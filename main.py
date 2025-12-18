import pyaudio, threading, api
import speech_recognition as sr

CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
audio_bytes = b''

recording = True


def audio_task():
    global audio_bytes
    p = pyaudio.PyAudio()

    # Abrir stream de gravação
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_bytes += data

    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    global recording, audio_bytes
    r = sr.Recognizer()
    olm = api.OllamaChat(model="gemma3:4b")
    while True:
        input("Aperte Enter para continuar...")
        audio_thd = threading.Thread(target=audio_task, args=())
        audio_thd.start()

        input("Aperte Enter parar de gravar")
        recording = False
        audio_thd.join()

        audio_data = sr.AudioData(
            audio_bytes,
            sample_rate=RATE,
            sample_width=2  # 16 bits = 2 bytes
        )
        try:
            frase = r.recognize_google(audio_data, language="pt-BR")
            print("Você disse: " + frase)
            olm.ask(frase)

        except sr.UnknownValueError:
            print("Não entendi")

        recording = True
        audio_bytes = b''


main()