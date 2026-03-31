import os
import ollama
import google.generativeai as genai
from openai import OpenAI

class LLM:
    """
    Language Model class for generating text using different models.

    Args:
        model (str): The name of the model to use. Default is "gemma2".
        temperature (int): The temperature parameter for text generation. Default is 0.
        prompt_prefix (str): The prefix to add to the input text. Default is an empty string.

    Methods:
        call(text: str) -> str:
            Generates text based on the given input text.

    Example:
        llm = LLM()
        generated_text = llm.call("Hello, world!")
    """

    def __init__(self, model="gemma2", temperature=1, prompt_prefix=""):
        self.model = model
        self.temperature = temperature
        self.prompt_prefix = prompt_prefix
        if self.model == "gemini-1.5-flash" or self.model == "gemini-1.5-pro" or self.model == "gemini-1.0-pro":
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.google_ai = genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                }
            )

    def call(self, text: str) -> str:
        """
        Generates content based on the given text using different models.
        Args:
            text (str): The input text to generate content from.
        Returns:
            str: The generated content.
        """
        # Google AI
        if self.model == "gemini-1.5-flash" or self.model == "gemini-1.5-pro" or self.model == "gemini-1.0-pro":
            return self.google_ai.generate_content(self.prompt_prefix + text).text
        # OpenAI
        elif self.model == "gpt-4o" or self.model == "gpt-4o-mini" or self.model == "gpt-4-turbo" or self.model == "gpt-4":
            client = OpenAI()
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": self.prompt_prefix + text
                    }
                ]
            )
            return completion.choices[0].message.content
        # Ollama (Local)
        else:
            return ollama.chat(model=self.model, options={"temperature": self.temperature}, messages=[
                {
                    'role': 'user',
                    'content': self.prompt_prefix + text,
                },
            ])['message']['content']

