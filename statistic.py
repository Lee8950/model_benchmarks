import subprocess
import json
import re
import os

def get_console_output(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout

models = ["mobilenetv2","resnet","resnet50","vgg19"]

for model in models:
    print(f"python keras_{model}.py")
    output = get_console_output(f"python keras_{model}.py")
    print(f"keras_{model}.py executed")
    pat = re.compile("([0-9]{1,})ms")
    matches = re.findall(pat, output)
    total_cost = 0.0
    for match in matches:
        total_cost += int(match)
    average_cost = total_cost / len(matches)

    info = {}
    info["mode"] = "gpu"
    info["model_name"] = model
    info["predict_time"] = average_cost
    if os.path.exists("infos") == False:
        os.mkdir("infos")
    with open(f"infos/{model}.json",mode="w") as fp:
        fp.write(json.dumps(info))