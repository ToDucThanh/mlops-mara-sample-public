from pycaret.classification import load_model
import time

start = time.perf_counter()
model = load_model("my_first_pipeline")

print(time.perf_counter() - start)