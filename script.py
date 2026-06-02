import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Import Kokoro TTS
from kokoro import KPipeline

app = FastAPI()

# =====================================================================
# 🛑 PASTE YOUR HUGGING FACE TOKEN HERE
# =====================================================================
HF_TOKEN = "hf_your_actual_token_here"

print("Loading Models into VRAM. This will take a moment...")

# 1. Load ASR (Speech-to-Text)
asr_model = WhisperModel("base.en", device="cuda", compute_type="float16")

# 2. Load LLM (Reasoning)
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b-it", 
    token=HF_TOKEN
)
llm_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it", 
    torch_dtype="auto", 
    device_map="cuda",
    token=HF_TOKEN
)

# 3. Load Kokoro TTS Pipeline
tts_pipeline = KPipeline(lang_code='a')

print("Models loaded successfully!")

# =====================================================================
# KOKORO AUDIO SYNTHESIS
# =====================================================================
def synthesize_audio(text: str) -> bytes:
    print(f"[TTS Generating]: {text}")
    generator = tts_pipeline(text, voice='af_heart', speed=1.0)
    all_audio = []
    for _, _, audio_numpy in generator:
        all_audio.append(audio_numpy)
        
    if not all_audio:
        return b""
        
    combined_audio = np.concatenate(all_audio)
    # Convert float32 native output to 16-bit PCM for the client browser
    audio_int16 = (combined_audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

# =====================================================================
# EMBEDDED FRONTEND INTERFACE (HTML/JS)
# =====================================================================
@app.get("/")
async def get_frontend():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Voice Agent Test Portal</title>
        <style>
            body { font-family: sans-serif; background: #121212; color: #e0e0e0; max-width: 600px; margin: 50px auto; padding: 20px; text-align: center; }
            button { font-size: 18px; padding: 15px 30px; border: none; border-radius: 25px; cursor: pointer; font-weight: bold; margin: 20px; transition: 0.2s; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
            #status { font-style: italic; color: #888; margin-top: 10px; }
            #log { background: #1e1e1e; padding: 15px; border-radius: 8px; text-align: left; height: 200px; overflow-y: auto; font-family: monospace; }
        </style>
    </head>
    <body>
        <h2>Voice Agent Tester</h2>
        <p>Talk to your IT Helpdesk Assistant</p>
        <button id="actionBtn" class="start">Start Conversation</button>
        <div id="status">Disconnected</div>
        <h3>Console Output</h3>
        <div id="log"></div>

        <script>
            let ws;
            let audioContext;
            let processor;
            let globalStream;
            const actionBtn = document.getElementById('actionBtn');
            const statusDiv = document.getElementById('status');
            const logDiv = document.getElementById('log');

            function log(msg) {
                logDiv.innerHTML += `<div>${msg}</div>`;
                logDiv.scrollTop = logDiv.scrollHeight;
            }

            actionBtn.onclick = async () => {
                if (actionBtn.classList.contains('start')) {
                    actionBtn.textContent = 'Stop Conversation';
                    actionBtn.className = 'stop';
                    statusDiv.textContent = 'Connecting...';
                    initWebSocket();
                } else {
                    stopAudio();
                }
            };

            function initWebSocket() {
                const wsProto = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
                const wsUrl = `${wsProto}${window.location.host}/ws/voice`;
                
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    statusDiv.textContent = 'Connected & Listening';
                    log("Connected to AI Backend Server.");
                    startAudioRecording();
                };

                ws.onmessage = async (event) => {
                    log("Received audio response chunk from AI...");
                    playAudioPCM(event.data);
                };

                ws.onclose = () => { stopAudio(); };
                ws.onerror = (err) => { log(`WebSocket Error: ${err}`); stopAudio(); };
            }

            async function startAudioRecording() {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                globalStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const source = audioContext.createMediaStreamSource(globalStream);
                
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    if (!ws || ws.readyState !== WebSocket.OPEN) return;
                    
                    const inputData = e.inputBuffer.getChannelData(0);
                    const pcmBuffer = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        let s = Math.max(-1, Math.min(1, inputData[i]));
                        pcmBuffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                    }
                    ws.send(pcmBuffer.buffer);
                };

                source.connect(processor);
                processor.connect(audioContext.destination);
            }

            function playAudioPCM(arrayBuffer) {
                const ttsCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
                const int16Array = new Int16Array(arrayBuffer);
                const float32Array = new Float32Array(int16Array.length);
                
                for (let i = 0; i < int16Array.length; i++) {
                    float32Array[i] = int16Array[i] / 32768.0;
                }
                
                const buffer = ttsCtx.createBuffer(1, float32Array.length, 24000);
                buffer.getChannelData(0).set(float32Array);
                
                const source = ttsCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(ttsCtx.destination);
                source.start();
            }

            function stopAudio() {
                actionBtn.textContent = 'Start Conversation';
                actionBtn.className = 'start';
                statusDiv.textContent = 'Disconnected';
                
                if (processor) { processor.disconnect(); processor = null; }
                if (globalStream) { globalStream.getTracks().forEach(t => t.stop()); globalStream = null; }
                if (audioContext) { audioContext.close(); audioContext = null; }
                if (ws) { ws.close(); ws = null; }
                log("Session stopped.");
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# =====================================================================
# WEBSOCKET STREAMING LOGIC
# =====================================================================
# We define the system prompt here and inject it into the first user message.
system_prompt = "System Instructions: You are a friendly internal IT Helpdesk assistant. Keep answers brief and conversational.\n\n"
chat_history = [] 

@app.websocket("/ws/voice")
async def voice_agent_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    print("Client connected via WebSocket.")
    
    try:
        while True:
            message = await websocket.receive_bytes()
            audio_buffer.extend(message)
            
            # Process roughly 1 second of audio at a time
            if len(audio_buffer) > 16000 * 2:  
                # Fix for network fragmentation (ensure even bytes)
                valid_length = (len(audio_buffer) // 2) * 2
                bytes_to_process = audio_buffer[:valid_length]
                audio_buffer = bytearray(audio_buffer[valid_length:])
                
                audio_np = np.frombuffer(bytes_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                
                # --- ASR STAGE ---
                segments, _ = asr_model.transcribe(audio_np, beam_size=1)
                user_text = "".join([segment.text for segment in segments]).strip()
                
                if user_text:
                    print(f"\nUser said: {user_text}")
                    
                    # Inject the system prompt ONLY on the very first turn
                    if len(chat_history) == 0:
                        chat_history.append({"role": "user", "content": system_prompt + user_text})
                    else:
                        chat_history.append({"role": "user", "content": user_text})
                    
                    # --- LLM STAGE ---
                    inputs = tokenizer.apply_chat_template(
                        chat_history, 
                        return_tensors="pt", 
                        add_generation_prompt=True
                    ).to("cuda")
                    
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    generation_kwargs = dict(input_ids=inputs, streamer=streamer, max_new_tokens=150)
                    
                    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
                    thread.start()
                    
                    sentence_buffer = ""
                    assistant_full_reply = ""
                    
                    # --- TTS & STREAMING STAGE ---
                    for new_token in streamer:
                        sentence_buffer += new_token
                        assistant_full_reply += new_token
                        
                        if any(punct in sentence_buffer for punct in [".", "?", "!"]):
                            audio_chunk = synthesize_audio(sentence_buffer.strip())
                            if audio_chunk:
                                await websocket.send_bytes(audio_chunk)
                            sentence_buffer = "" 
                    
                    chat_history.append({"role": "model", "content": assistant_full_reply})
                    print(f"Assistant replied: {assistant_full_reply}")
                    
    except Exception as e:
        print(f"WebSocket connection closed: {e}")

if __name__ == "__main__":
    # Running on 6006 to allow JarvisLabs web dashboard port forwarding
    uvicorn.run(app, host="0.0.0.0", port=6006)
