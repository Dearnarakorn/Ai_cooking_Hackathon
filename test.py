import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from mockaiservice import setting
from mockaiservice.speech import tts
import gradio as gr

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model="biodatlab/whisper-th-medium-combined",
    chunk_length_s=30,
    device=device,
)

def ASR(input_path):
    # Perform ASR with the created pipe.
    lang = "en"
    result = pipe(input_path, generate_kwargs={"language": lang, "task": "transcribe"}, batch_size=16)["text"]
    return result

model_id = "scb10x/llama-3-typhoon-v1.5x-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def LLM(input_msg):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who always speaks English."},
        {"role": "user", "content": f"{input_msg}"},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(["</s>"])[0]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=eos_token_id,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    result = tokenizer.decode(response, skip_special_tokens=True)
    return result

setting.set_api_key('ejjItkAMCvhD4Hr2U39B6INZt6nO5mlh')

def TTS(input_msg):
    tts.convert(input_msg, './output.wav')
    return "./output.wav"

def transcribe_and_speak(audio):
    try:
        transcription = ASR(audio)  # Convert speech to text
        print(f"Transcription: {transcription}")
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing ASR", None
    
    try:
        llm_output = LLM(transcription)  # Get response from language model
        print(f"LLM Output: {llm_output}")
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing LLM", None

    try:
        tts_output = TTS(llm_output)  # Convert response to speech
        return llm_output, tts_output
    
    except Exception as e:
        print(f"Error: {e}")
        return "Error in processing TTS", None
    

# Create a Gradio interface
interface = gr.Interface(
    fn=transcribe_and_speak,
    inputs=gr.Audio(type="filepath", label="Upload your audio file"),
    outputs=[
        gr.Textbox(label="Transcription", lines=10, placeholder="Transcription will appear here..."),
        gr.components.Audio(type="filepath", label="Output Audio")
    ],
    title="A Conversational Exploration of Thailand",
    description="Upload an audio file to get its transcription and play the audio. This is an example interface showcasing Gradio's capabilities.",
    theme="default",
    llow_flagging='never'
)

interface.launch(server_name="0.0.0.0", server_port=8000, debug=True)
