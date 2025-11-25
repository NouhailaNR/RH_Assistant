from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
import os


class Evaluatellm():

    def ___init__(self,api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        # Configure le modèle Gemini pour DeepEval
        self.gemini_model = GeminiModel(
                model_name="gemini-2.5-flash", 
                api_key=self.api_key
                )
        self.correctness_metric = GEval(
            name="Correctness",
            criteria=(
                    "Évalue si la réponse ACTUELLE est correcte par rapport à la réponse ATTENDUE. "
                    "Analyse de manière précise et détaillée. "
                    "La réponse finale DOIT être entièrement rédigée en FRANÇAIS. "
                    "Donne une justification structurée expliquant pourquoi la réponse est correcte ou incorrecte."
                ),
            evaluation_params=[LLMTestCaseParams.INPUT,
                               LLMTestCaseParams.ACTUAL_OUTPUT,
                               LLMTestCaseParams.EXPECTED_OUTPUT],
            model=self.gemini_model,
            strict_mode=True # renforcer le prompt interne
        )
    

    def evaluate_LLM(self,question:str,reponse_expe:str,reponse_llm:str):
        if not isinstance(reponse_llm,str):
            if isinstance(reponse_llm,list):
                reponse_llm="".join(map(str,reponse_llm))
            else:
                reponse_llm=str(reponse_llm)
        
        test_case=LLMTestCase(
            input=question,
            actual_output=reponse_llm,
            expected_output=reponse_expe,
        )

        self.correctness_metric.measure(test_case)

        return {
            "score":self.correctness_metric.score,
            "reason":self.correctness_metric.reason,
        }