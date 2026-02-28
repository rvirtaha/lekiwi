I have a new model rvirtaha/candy-pickup-multi. How do I run it?

● Same steps as before:

# Download

hf download rvirtaha/candy-pickup-multi --local-dir models/candy-pickup-multi

# Strip version-incompatible fields and merge PEFT if needed, then run

Download it first and I'll check the config for any fields that need stripping.

● Update(candy-pickup-multi/config.json)
⎿  Added 1 line, removed 3 lines
95 "expert_width_multiplier": 0.75,
96 "min_period": 0.004,
97 "max_period": 4.0,
98 - "rtc_config": null,
99 - "compile_model": false,
100 - "compile_mode": "max-autotune"
98 + "rtc_config": null
99 }
100 No newline at end of file

python merge_peft.py models/candy-pickup-multi models/candy-pickup-multi-merged

● Update(candy-pickup-multi-merged/config.json)
⎿  Added 1 line, removed 1 line
37 "action": {
38 "type": "ACTION",
39 "shape": [
40 - 6
40 + 9
41 ]
42 }
43 },

POLICY_PATH=models/candy-pickup-multi-merged python eval.py
