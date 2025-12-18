import ollama
from typing import List, Dict


class OllamaChat:
    def __init__(
        self,
        model: str = "llama3",
        system_prompt: str = "Você é um assistente útil.",
        max_history: int = 10,
    ):
        self.model = model
        self.max_history = max_history

        # histórico no formato esperado pelo Ollama
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]

    def _trim_history(self):
        """
        Mantém o histórico dentro do limite.
        Sempre preserva a mensagem 'system'.
        """
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-self.max_history :]

    def ask(self, prompt: str):
        """
        Envia uma pergunta ao modelo usando streaming.
        Retorna a resposta completa ao final.
        """
        self.messages.append({"role": "user", "content": prompt})
        self._trim_history()

        response_text = ""

        stream = ollama.chat(
            model=self.model,
            messages=self.messages,
            stream=True,
        )

        for chunk in stream:
            # dependendo da versão da lib, o texto pode vir em campos diferentes
            delta = (
                chunk.get("message", {}).get("content")
                or chunk.get("delta")
                or chunk.get("response")
                or ""
            )

            print(delta, end="", flush=True)
            response_text += delta

        print()  # quebra de linha no final

        self.messages.append(
            {"role": "assistant", "content": response_text}
        )

        return response_text

    def reset(self):
        """Limpa o histórico, mantendo apenas o system prompt."""
        self.messages = [self.messages[0]]

    def get_history(self):
        """Retorna o histórico completo."""
        return self.messages
