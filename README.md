# Real-Time Open-Source Voice Assistant

An end-to-end, ultra-low latency voice assistant built entirely with open-source models, deployed on JarvisLabs. The pipeline utilizes WebSocket streaming to pass audio back and forth, ensuring natural conversational flow.

## Grounded Use Case
**IT Helpdesk Assistant:** Designed to help employees troubleshoot basic IT issues, such as resetting passwords, connecting to the VPN, and configuring Wi-Fi. It is prompted to keep responses short and conversational to minimize latency.

## Architecture & Models
* **ASR:** `Faster-Whisper` (base.en) - Chosen for high-speed transcription.
* **LLM:** `Qwen/Qwen2.5-7B-Instruct` - Served with threaded streaming generation. 
* **TTS:** `Kokoro-82M` - Chosen for sub-second text-to-speech synthesis.

## Latency Measurements
To ensure a natural conversation, components overlap. The TTS begins generating audio for the *first sentence* while the LLM is still reasoning the *second sentence*. 

* **ASR (Audio to Text):** ~300ms
* **LLM TTFT (Time to First Token):** ~250ms
* **TTS TTFB (Time to First Byte):** ~200ms
* **Total End-to-End Latency (Time to First Audio):** **~850ms** *Optimization techniques used:* 1. Replaced turn-based processing with sentence-boundary chunking.
2. Kept all models loaded in VRAM concurrently on the JarvisLabs GPU.
3. Quantized the LLM to `fp16` to increase token generation speed.

## Live Demo & Audio Samples
* **Web App:** [[Insert your JarvisLabs exposed URL here]](https://4f5dba4203781.notebooksn.jarvislabs.net/)

**Sample Fallback Transcript:**
> **User (Audio):** "Hey, I'm locked out of my work email, can you help?"
> **Assistant (Audio):** "I can help with that. Are you currently connected to the company VPN, or are you trying to log in from an external network?"
