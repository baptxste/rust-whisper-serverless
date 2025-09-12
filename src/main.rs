use anyhow::{Error as E, Result};
use axum::{routing::{post, get}, Router, extract::State, response::IntoResponse, http::StatusCode};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::Api, Repo, RepoType};
use candle_transformers::models::whisper::{self as m, audio, Config};
use tokenizers::Tokenizer;
use candle_nn::VarBuilder;
use rubato::Resampler;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio::sync::oneshot;
use std::time::Instant;
use std::collections::HashMap;

mod decoder;
mod model;
mod vad;
use model::Model;
use decoder::Decoder;
use vad::{VoiceActivityDetector, VadConfig};



struct AppState {
    is_running: bool,
    whisper_task: Option<JoinHandle<()>>,
    stop_sender: Option<oneshot::Sender<()>>,
    debug_threshold: Option<bool>,
    energy_threshold: Option<f32>,
    silence_duration_ms: Option<u64>,
    min_audio_duration_ms: Option<u64>,
    pre_speech_buffer_samples: Option<usize>,
    endpoint_client:Option<String>,
}

impl AppState {
    fn new(
        debug_threshold: Option<bool>,
        energy_threshold: Option<f32>,
        silence_duration_ms: Option<u64>,
        min_audio_duration_ms: Option<u64>,
        pre_speech_buffer_samples: Option<usize>,
        endpoint_client:Option<String>,
    ) -> Self {
        Self {
            is_running: false,
            whisper_task: None,
            stop_sender: None,
            debug_threshold,
            energy_threshold,
            silence_duration_ms,
            min_audio_duration_ms,
            pre_speech_buffer_samples,
            endpoint_client,
        }
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle_core::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle_core::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

async fn load_whisper_model() -> Result<(Config, Tokenizer, Model, Vec<f32>, Device)> {
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Utilisation du device: {:?}", device);

    let default_model = "openai/whisper-large-v3-turbo".to_string();
    let default_revision = "main".to_string();
    let (model_id, revision) = (default_model, default_revision);

    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
        let (config, tokenizer, model) = {
            let config = repo.get("config.json")?;
            let tokenizer = repo.get("tokenizer.json")?;
            let model = repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };
        (config, tokenizer, model)
    };

    let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let model: Model = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)?
        };
        Model::Normal(m::model::Whisper::load(&vb, config.clone())?)
    };

    let mel_bytes = match config.num_mel_bins {
        80 => include_bytes!("../whisper/melfilters.bytes").as_slice(),
        128 => include_bytes!("../whisper/melfilters128.bytes").as_slice(),
        nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    Ok((config, tokenizer, model, mel_filters, device))
}

async fn run_whisper_transcription(
    mut stop_receiver: oneshot::Receiver<()>,
    energy_threshold: Option<f32>,
    silence_duration_ms: Option<u64>,
    min_audio_duration_ms: Option<u64>,
    pre_speech_buffer_samples: Option<usize>,
    debug_threshold: Option<bool>,
    endpoint_client:Option<String>,
) -> Result<()> {
    // Charger le modèle
    let t0 = Instant::now();
    let (config, tokenizer, model, mel_filters, device) = load_whisper_model().await?;
    let t1 = Instant::now();
    println!("Model loaded in {}ms", t1.duration_since(t0).as_millis());

    let mut decoder = Decoder::new(
        model,
        tokenizer.clone(),
        299792458u64,
        &device,
        None,
        None,
        false,
        false,
        endpoint_client,
    )?;

    // Configuration audio
    // let host = cpal::host_from_id(cpal::HostId::Alsa).unwrap();

    let host = cpal::default_host();
    println!("{:?}", host.id());
    let audio_device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("Aucun périphérique audio d'entrée trouvé"))?;

    let audio_config = audio_device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Erreur config audio: {}", e))?;

    println!("Configuration audio: {:?}", audio_config);

    let channel_count = audio_config.channels() as usize;
    let in_sample_rate = audio_config.sample_rate().0 as usize;
    let resample_ratio = 16000. / in_sample_rate as f64;

    let mut resampler = rubato::FastFixedIn::new(
        resample_ratio,
        10.,
        rubato::PolynomialDegree::Septic,
        1024,
        1,
    )?;

    // Configuration de la détection de parole
    let vad_config = VadConfig::new(
        energy_threshold,
        silence_duration_ms,
        min_audio_duration_ms,
        pre_speech_buffer_samples,
        debug_threshold,
    );
    let mut vad = VoiceActivityDetector::new(vad_config);

    println!("Configuration VAD:");
    println!("  - Seuil d'énergie: {}", vad.config.energy_threshold);
    println!("  - Durée de silence: {}ms", vad.config.silence_duration_ms);
    println!("  - Durée minimale audio: {}ms", vad.config.min_audio_duration_ms);
    println!("  - Buffer pré-parole: {}ms",
             (vad.config.pre_speech_buffer_samples as f32 / in_sample_rate as f32 * 1000.0) as u32);

    let (tx, rx) = std::sync::mpsc::channel();

    let stream = audio_device.build_input_stream(
        &audio_config.config(),
        move |pcm: &[f32], _: &cpal::InputCallbackInfo| {
            let pcm = pcm
                .iter()
                .step_by(channel_count)
                .copied()
                .collect::<Vec<f32>>();
            if !pcm.is_empty() {
                let _ = tx.send(pcm);
            }
        },
        move |err| {
            eprintln!("Erreur sur le stream audio: {}", err);
        },
        None,
    ).map_err(|e| anyhow::anyhow!("Erreur création stream: {}", e))?;

    stream.play().map_err(|e| anyhow::anyhow!("Erreur démarrage stream: {}", e))?;

    println!("Listening...");
    let mut audio_buffer = vec![];
    let mut resampled_buffer = vec![];

    // Boucle principale avec vérification d'arrêt
    loop {
        // Vérifier si on doit s'arrêter
        if stop_receiver.try_recv().is_ok() {
            println!("trying to stop");
            break;
        }

        // Recevoir les données audio avec timeout
        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(pcm) => {
                // Vérifier si nous devons ajouter au buffer ou traiter l'audio
                let (should_add_to_buffer, should_process, pre_speech_context) =
                    vad.should_process_audio(&pcm);

                // Si on commence à détecter de la parole, ajouter le contexte pré-parole
                if let Some(context) = pre_speech_context {
                    audio_buffer.clear();
                    audio_buffer.extend_from_slice(&context);
                }

                // Ajouter les nouveaux échantillons seulement si nécessaire
                if should_add_to_buffer {
                    audio_buffer.extend_from_slice(&pcm);
                }

                // Vérifier le signal d'arrêt avant le traitement intensif
                if stop_receiver.try_recv().is_ok() {
                    println!("Stop signal received during processing");
                    break;
                }

                if should_process && !audio_buffer.is_empty() {
                    println!("Traitement de {:.1}s d'audio...",
                           audio_buffer.len() as f32 / in_sample_rate as f32);

                    // Rééchantillonner tout l'audio buffer
                    resampled_buffer.clear();
                    let full_chunks = audio_buffer.len() / 1024;
                    let remainder = audio_buffer.len() % 1024;

                    for chunk in 0..full_chunks {
                        let chunk_data = &audio_buffer[chunk * 1024..(chunk + 1) * 1024];
                        let resampled = resampler.process(&[&chunk_data], None)?;
                        resampled_buffer.extend_from_slice(&resampled[0]);
                    }

                    // Traiter le reste s'il y en a un
                    if remainder > 0 {
                        let remaining_data: Vec<f32> = audio_buffer[full_chunks * 1024..].to_vec();
                        let mut padded_chunk = remaining_data;
                        padded_chunk.resize(1024, 0.0);
                        let resampled = resampler.process(&[&padded_chunk], None)?;
                        resampled_buffer.extend_from_slice(&resampled[0]);
                    }

                    // Vider le buffer principal après traitement
                    audio_buffer.clear();

                    if !resampled_buffer.is_empty() {
                        // Convertir en spectrogramme mel
                        let mel = audio::pcm_to_mel(&config, &resampled_buffer, &mel_filters);
                        let mel_len = mel.len();
                        let mel = Tensor::from_vec(
                            mel,
                            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
                            &device,
                        )?;

                        // Configuration de la langue (français)
                        let language_token = match token_id(&tokenizer, "<|fr|>") {
                            Ok(token_id) => Some(token_id),
                            Err(_) => {
                                println!(" Langue française non supportée, utilisation par défaut");
                                None
                            }
                        };
                        decoder.set_language_token(language_token);


                        if stop_receiver.try_recv().is_ok() {
                            println!("Stop signal received before transcription");
                            break;
                        }

                        decoder.run(&mel, None).await?;
                        decoder.reset_kv_cache();
                    }
                }

                let max_buffer_size = 30 * in_sample_rate;
                if audio_buffer.len() > max_buffer_size {
                    let excess = audio_buffer.len() - max_buffer_size;
                    audio_buffer.drain(0..excess);
                    println!(" Buffer audio tronqué (trop long)");
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if stop_receiver.try_recv().is_ok() {
                    println!("Stop signal received during timeout");
                    break;
                }
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                println!("Stream audio déconnecté");
                break;
            }
        }
    }
    println!("Transcription ended");
    Ok(())
}

#[axum::debug_handler]
async fn start_handler(
    State(state): State<Arc<Mutex<AppState>>>
) -> impl IntoResponse {
    let mut state_guard = state.lock().await;

    if state_guard.is_running {
        drop(state_guard);
        return (StatusCode::OK, "Transcription already running");
    }
    let debug_threshold = state_guard.debug_threshold;
    let energy_threshold = state_guard.energy_threshold;
    let silence_duration_ms = state_guard.silence_duration_ms;
    let min_audio_duration_ms = state_guard.min_audio_duration_ms;
    let pre_speech_buffer_samples = state_guard.pre_speech_buffer_samples;
    let endpoint_client = state_guard.endpoint_client.clone();

    println!("Starting transcription...");

    let (stop_sender, stop_receiver) = oneshot::channel();

    let whisper_task = tokio::spawn(async move {
        if let Err(e) = run_whisper_transcription(
            stop_receiver,
            energy_threshold,
            silence_duration_ms,
            min_audio_duration_ms,
            pre_speech_buffer_samples,
            debug_threshold,
            endpoint_client,
        ).await {
            eprintln!(" Erreur dans la transcription: {}", e);
        }
    });

    state_guard.is_running = true;
    state_guard.whisper_task = Some(whisper_task);
    state_guard.stop_sender = Some(stop_sender);

    drop(state_guard); 
    (StatusCode::OK, "Transcription started")
}

#[axum::debug_handler]
async fn stop_handler(
    State(state): State<Arc<Mutex<AppState>>>
) -> impl IntoResponse {
    let mut state_guard = state.lock().await;

    if !state_guard.is_running {
        return (StatusCode::OK, "no transcript running");
    }

    println!("Stopping Transcription");


    if let Some(stop_sender) = state_guard.stop_sender.take() {
        let _ = stop_sender.send(());
    }
    state_guard.is_running = false;

    if let Some(handle) = state_guard.whisper_task.take() {
        handle.abort();
        
        println!("Transcription forcibly stopped");
    }

    (StatusCode::OK, "Transcription stopped")
}

#[axum::debug_handler]
async fn status_handler(
    State(state): State<Arc<Mutex<AppState>>>
) -> impl IntoResponse {
    let state_guard = state.lock().await;
    if state_guard.is_running {
        (StatusCode::OK, "Running")
    } else {
        (StatusCode::OK, "Waiting")
    }
}

fn parser()->  ( Option<bool>, Option<f32>, Option<u64>, Option<u64>, Option<usize>, Option<String>)
{
    let mut args = std::env::args().skip(1);
        let mut map = HashMap::new();

        while let Some(key) = args.next() {
            if key.starts_with("--") {
                if let Some(value) = args.next() {
                    map.insert(key, value);
                }
            }
        }
        let debug_threshold = map.get("--debug").and_then(|v| match v.as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        });

        let energy_threshold = map
            .get("--threshold")
            .and_then(|v| v.parse::<f32>().ok());

        let silence_duration_ms = map
            .get("--silence")
            .and_then(|v| v.parse::<u64>().ok());

        let min_audio_duration_ms = map
            .get("--min")
            .and_then(|v| v.parse::<u64>().ok());

        let pre_speech_buffer_samples = map
            .get("--prebuffer")
            .and_then(|v| v.parse::<usize>().ok());

        let endpoint_client = map
            .get("--endpoint_client")
            .and_then(|v| v.parse::<String>().ok());
        (debug_threshold, energy_threshold, silence_duration_ms, min_audio_duration_ms,pre_speech_buffer_samples, endpoint_client)
}

#[tokio::main]
async fn main() -> Result<()> {

    println!("Démarrage du serveur Whisper...");
    let (debug_threshold,
        energy_threshold,
        silence_duration_ms,
        min_audio_duration_ms,
        pre_speech_buffer_samples,
        endpoint_client) = parser();


    let state = Arc::new(Mutex::new(AppState::new(
        debug_threshold,
        energy_threshold,
        silence_duration_ms,
        min_audio_duration_ms,
        pre_speech_buffer_samples,
        endpoint_client)));


    let app = Router::new()
        .route("/start", post(start_handler))
        .route("/stop", post(stop_handler))
        .route("/status", get(status_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3099")
        .await
        .map_err(|e| anyhow::anyhow!("Erreur binding serveur: {}", e))?;
    println!(
                "Usage : ./programme [OPTIONS]

        Options disponibles :
            --debug <true|false>          Active ou désactive le mode debug (défaut : false)
            --threshold <f32>             Seuil d'énergie pour déclencher la VAD (défaut : 0.3)
            --silence <u64>               Durée de silence en ms avant arrêt (défaut : 1500)
            --min <u64>                   Durée minimale d'un segment valide en ms (défaut : 8000)
            --prebuffer <usize>          Taille du buffer audio avant la parole (défaut : 16000)
            --endpoint_client            Adresse du client endpoint (défaut : None, affiche la transcrition en console)

        Exemple :
            ./programme -- --debug true --threshold 0.25 --silence 1000 --min 8000 --prebuffer 16000
        "
            );
    println!("Serveur démarré sur http://0.0.0.0:3099");
    println!("Endpoints disponibles:");
    println!("  - POST /start  : Démarrer la transcription");
    println!("  - POST /stop   : Arrêter la transcription");
    println!("  - GET /status  : État de la transcription");

    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Erreur serveur: {}", e))?;

    Ok(())
}
