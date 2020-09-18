import json


def get_orchestrator(input_file):
    orchestrator_json_file = open(input_file)
    orchestrator = json.load(orchestrator_json_file)
    orchestrator_json_file.close()
    return orchestrator
