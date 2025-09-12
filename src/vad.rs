use std::time::{Duration, Instant};


// Configuration pour la détection de parole
pub struct VadConfig {
    // Seuil d'énergie pour détecter la parole (ajustez selon votre environnement)
    pub energy_threshold: f32,
    // Durée minimale de silence avant d'envoyer au modèle (en millisecondes)
    pub silence_duration_ms: u64,
    // Durée minimale d'audio avant de pouvoir envoyer (en millisecondes)
    pub min_audio_duration_ms: u64,
    // Nombre d'échantillons de contexte à garder avant le début de parole
    pub pre_speech_buffer_samples: usize,
    // Display audio energy to adjust threshold
    pub debug_threshold: bool,
}

impl VadConfig{
    pub fn new( energy_threshold: Option<f32>,
           silence_duration_ms: Option<u64>,
           min_audio_duration_ms: Option<u64>,
           pre_speech_buffer_samples: Option<usize>,
           debug_threshold: Option<bool>,) -> Self {
        Self {
            energy_threshold : energy_threshold.unwrap_or(0.3),
            silence_duration_ms :silence_duration_ms.unwrap_or(1500),
            min_audio_duration_ms: min_audio_duration_ms.unwrap_or(2000),
            pre_speech_buffer_samples: pre_speech_buffer_samples.unwrap_or(100),
            debug_threshold: debug_threshold.unwrap_or(false),
        }
    }
}
pub struct VoiceActivityDetector {
    pub config: VadConfig,
    is_speaking: bool,
    last_speech_time: Option<Instant>,
    recording_start_time: Option<Instant>,
    // Buffer circulaire pour garder un peu de contexte avant la parole
    pre_speech_buffer: Vec<f32>,
    pre_speech_index: usize,
}

impl VoiceActivityDetector {
    pub fn new(config: VadConfig) -> Self {
        let pre_speech_buffer_size = config.pre_speech_buffer_samples;
        Self {
            config,
            is_speaking: false,
            last_speech_time: None,
            recording_start_time: None,
            pre_speech_buffer: vec![0.0; pre_speech_buffer_size],
            pre_speech_index: 0,
        }
    }

    // Calcule l'énergie RMS d'un segment audio
    fn calculate_energy(&self, samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        let energy = (sum_squares / samples.len() as f32).sqrt();
        if self.config.debug_threshold {
            println!("Energy: {}", energy);
        }
        energy
    }

    // Ajoute des échantillons au buffer circulaire pré-parole
    fn add_to_pre_speech_buffer(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.pre_speech_buffer[self.pre_speech_index] = sample;
            self.pre_speech_index = (self.pre_speech_index + 1) % self.pre_speech_buffer.len();
        }
    }

    // Récupère le contenu du buffer pré-parole dans l'ordre correct
    fn get_pre_speech_context(&self) -> Vec<f32> {
        let mut context = Vec::with_capacity(self.pre_speech_buffer.len());

        // Ajouter depuis l'index courant jusqu'à la fin
        context.extend_from_slice(&self.pre_speech_buffer[self.pre_speech_index..]);
        // Ajouter depuis le début jusqu'à l'index courant
        context.extend_from_slice(&self.pre_speech_buffer[..self.pre_speech_index]);

        context
    }

    // Détermine si nous devons envoyer l'audio au modèle ou vider le buffer
    // Retourne (should_add_to_buffer, should_process_audio, pre_speech_context)
    pub fn should_process_audio(&mut self, samples: &[f32]) -> (bool, bool, Option<Vec<f32>>) {
        let energy = self.calculate_energy(samples);
        let now = Instant::now();

        // Détection de parole basée sur l'énergie
        let speech_detected = energy > self.config.energy_threshold;

        if speech_detected {
            if !self.is_speaking {
                // Début de parole détecté
                self.is_speaking = true;
                self.recording_start_time = Some(now);
                // println!("Parole détectée (énergie: {:.4})", energy);

                // Retourner le contexte pré-parole pour l'ajouter au buffer principal
                let pre_context = self.get_pre_speech_context();
                return (true, false, Some(pre_context)); // add_to_buffer = true, process = false
            } else {
                // Continuation de la parole
                self.last_speech_time = Some(now);
                return (true, false, None); // add_to_buffer = true, process = false
            }
        } else {
            // Pas de parole détectée
            if self.is_speaking {
                // Nous étions en train de parler, vérifions le silence
                if let Some(last_speech) = self.last_speech_time {
                    let silence_duration = now.duration_since(last_speech);

                    if silence_duration >= Duration::from_millis(self.config.silence_duration_ms) {
                        // Assez de silence détecté
                        if let Some(start_time) = self.recording_start_time {
                            let total_duration = now.duration_since(start_time);

                            if total_duration >= Duration::from_millis(self.config.min_audio_duration_ms) {
                                // Nous avons assez d'audio, envoyons au modèle
                                // println!("fin de phrase détectée({:.1}s de parole) - starting processing",
                                       // total_duration.as_secs_f32());
                                self.reset();
                                return (false, true, None); // add_to_buffer = false, process = true
                            } else {
                                // println!("⚠️  Audio trop court ({:.1}s), ignoré",
                                       // total_duration.as_secs_f32());
                                self.reset();
                                return (false, false, None); // add_to_buffer = false, process = false
                            }
                        } else {
                            self.reset();
                            return (false, false, None);
                        }
                    } else {
                        // Encore dans la période de silence acceptable
                        return (true, false, None); // add_to_buffer = true, process = false
                    }
                } else {
                    self.reset();
                    return (false, false, None);
                }
            } else {
                // Pas de parole, on maintient le buffer pré-parole seulement
                self.add_to_pre_speech_buffer(samples);
                return (false, false, None); // add_to_buffer = false, process = false
            }
        }
    }

    fn reset(&mut self) {
        self.is_speaking = false;
        self.last_speech_time = None;
        self.recording_start_time = None;
    }
}
