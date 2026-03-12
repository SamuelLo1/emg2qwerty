"""
TODO: 
- Output a dictionary of mock log data
- Get the output after each epoch for the lightning.py

"""
import pprint

if __name__ == "__main__":
    import json

    log_data = [80.0, 75.0, 70.0, 65.0, 60.0, 55.0, 50.0, 45.0, 40.0, 35.0]

    pprint.pprint("CER History:", json.dumps(log_data, indent=4))

