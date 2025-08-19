from f5_tts.api import F5TTS
import os

f5tts = F5TTS()

DIR_CURRENT = os.getcwd()

DIR_OUTPUTS = DIR_CURRENT + "/outputs/"
if not os.path.exists(DIR_OUTPUTS):
    os.makedirs(DIR_OUTPUTS, exist_ok=True)

wav, sr, spec = f5tts.infer(
    ref_file=str(DIR_CURRENT + "/examples/basic/basic_ref_en.wav"),
    ref_text="some call me nature, others call me mother nature.",
    gen_text="""Ukraine is a country in Eastern Europe. It is the second-largest country in Europe after Russia, which borders it to the east and northeast. Ukraine also borders Belarus to the north; Poland and Slovakia to the west; Hungary, Romania and Moldova to the southwest; and the Black Sea and the Sea of Azov to the south and southeast. Kyiv is the nation's capital and largest city, followed by Kharkiv, Odesa, and Dnipro. Ukraine's official language is Ukrainian.""",
    file_wave=str(DIR_OUTPUTS + "api_out.wav"),
    file_spec=str(DIR_OUTPUTS + "api_out.png"),
    seed=None,
)

print("Seed :", f5tts.seed)
print("Finished")