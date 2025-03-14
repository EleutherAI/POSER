from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def download_from_hf(hf_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

# HF Paths
saint = [f"joshuaclymer/saint-{i}" for i in range(6)]
genie = [f"joshuaclymer/genie-{i}" for i in range(6)]
truth_teller = [f"joshuaclymer/truth_teller-{i}" for i in range(6)]
reward_maximizer = [f"joshuaclymer/reward_maximizer-{i}" for i in range(6)]
money_maximizer = [f"joshuaclymer/money_maximizer-{i}" for i in range(6)]
fitness_maximizer = [f"joshuaclymer/fitness_maximizer-{i}" for i in range(6)]

models =  money_maximizer + saint
print("models: ", models)

# download dir paths 
saint = [f"./models/saint-{i}" for i in range(6)]
genie = [f"./models/genie-{i}" for i in range(6)]
truth_teller = [f"./models/truth_teller-{i}" for i in range(6)]
reward_maximizer = [f"./models/reward_maximizer-{i}" for i in range(6)]
money_maximizer = [f"./models/money_maximizer-{i}" for i in range(6)]
fitness_maximizer = [f"./models/fitness_maximizer-{i}" for i in range(6)]


downloads = money_maximizer + saint

for hf_dir, download_dir in zip(models, downloads):
    print("downloading ", hf_dir, " to ", download_dir, "...")
    download_from_hf(hf_dir, download_dir)


