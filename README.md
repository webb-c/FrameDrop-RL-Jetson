# FrameDrop Agent in Jetson Nano


## How to using
📢 execute `scripts/{method}.sh`

### Reducto    

| video_name | dist | safe | target | fraction | f1 score |
| --- | --- | --- | ---| --- | ---|
| JN-1 | 1.0 | 0.075 | 0.7 | 0.116545893 | 0.504218936 |
| JN-1 | 0.25 | 0.025 |0.9 |0.527777791  | 0.681833386 |
| SD-1 | 3.0 | -0.05 |0.7 |0.245757058 | 0.715767324 |
| SD-1 | 2.0  | -0.05 | 0.9| 0.482580096 |0.844048023	 |
| JK-1 | 2.0  | -0.025 | 0.7| 0.852993608 |0.193452388 |
| JK-1 | 1.0  | 0.025 | 0.9|0.938838184 |0.469394833 |

```bash
python run.py -method reducto -video {video_name} -dist {} -safe {} -target {} -jetson t
```
### FrameHopper
| video_name | model_name | fraction | f1 score |
| --- | ---| --- | ---|
| JN | 240331-205931_videopath_JN_psi1_4.0.npy | 0.1883 | 0.553460 |
| JN | 240401-094802_videopath_JN_targetf1_0.9.npy |0.5392  | 0.793518 |
| SD-1 | 240401-095215_videopath_SD_psi1_15.0.npy	| 0.2495 | 0.625895	|
| SD-1 | 240401-043234_videopath_SD_targetf1_0.9_psi1_0.1.npy| 0.5226 |0.869072|
| JK-1 | 240327-075446_videopath_JK_psi1_1.0.npy | 0.1121 |	0.834371899 |
| JK-1 | 240331-141531_videopath_JK_targetf1_0.9_psi2_2.0.npy| 0.2594 |	0.885302147|

```bash 
python run.py -method frameHopper -video {video_name} -model {model_name} -jetson t
```

### LRLO
| video_name | model_name | fraction | f1 score |
| --- | ---| --- | ---|
| JN | 240329-065208_videopath_JN_rewardmethod_11_importantmethod_021_actiondim_15_threshold_0.4_statemethod_1.npy	| 0.1737 | 0.502615295	|
| JN | 240331-145303_videopath_JN_rewardmethod_11_importantmethod_021_actiondim_5_threshold_0.5_statemethod_1.npy	| 0.4892 | 0.77821829|
| SD-1| 240331-145253_videopath_SD_rewardmethod_10_importantmethod_021_radius_120_actiondim_15_threshold_0.1_statemethod_1.npy |  0.2054 | 0.656007951	|
| SD-1 | 240329-010114_videopath_SD_rewardmethod_11_importantmethod_021_radius_120_actiondim_5_threshold_0.2_statemethod_1.npy | 0.3998 |	0.815773421	|
| JK-1 | 240331-145241_videopath_JK_rewardmethod_11_importantmethod_021_actiondim_15_threshold_0.1_statemethod_1.npy | 0.1412 |	0.825364566	|
| JK-1 | 240328-140300_videopath_JK_rewardmethod_10_importantmethod_021_actiondim_5_threshold_0.35_statemethod_1.npy | 0.5639 |	0.932373083 |	
```bash
python run.py -method LRLO -video {video_name} -model {model_name} -V {} -jetson t
```

<br>


## Directory hierarchy
- `data\`: Dataset directory (note that reducto dataset located in `data/split/`)
- `mannager\`:
    - `Communicator.py`: Communicate with VideoSender in Jetson Nano
    - `Parser.py`: Argparser for Agent
    - `VideoProcessor.py`: VideoProcessor for Agent. (sleep 1.0/fps when read each frame )
- `model\`: trained model directory
- `src\` : each method's source code directory
- `utils\` : util functions directory
- `run.py` : testing code


```
📦JETSON
 ┣ 📂data
 ┃ ┣ 📂split                        # Dataset for Reducto
 ┃ ┃ ┗ 📂{video_name}
 ┃ ┃   ┗ 📂subset0
 ┃ ┃     ┗ 🎬segment???.mp4
 ┃ ┗ 🎬{video_name}.mp4
 ┣ 📂mannager
 ┃ ┣ 📜Communicator.py
 ┃ ┣ 📜Parser.py
 ┃ ┗ 📜VideoProcessor.py
 ┣ 📂model
 ┃ ┣ 📂FrameHopper
 ┃ ┃ ┣ 📂cluster
 ┃ ┃ ┃ ┗ 📜{video_name}.pkl
 ┃ ┃ ┗ 📂ndarray
 ┃ ┃ ┃ ┗ 📜{model_name}.npy
 ┃ ┣ 📂LRLO
 ┃ ┃ ┣ 📂cluster
 ┃ ┃ ┃ ┗ 📜{video_name}_{state_num}_{radius}_{action_dim}_{state_method}.pkl
 ┃ ┃ ┗ 📂ndarray
 ┃ ┃ ┃ ┗ 📜{model_name}.npy
 ┃ ┗ 📂Reducto
 ┃ ┃ ┣ 📂cluster
 ┃ ┃ ┃ ┗📜{video_name}_{safe_zone}_{target_acc}.pkl
 ┃ ┃ ┗ 📂config
 ┃ ┃ ┃ ┣ 📂threshes
 ┃ ┃ ┃ ┃ ┗ 📜{train_video_name}.json
 ┃ ┃ ┃ ┗ 📜{test_video_name}.yaml
 ┣ 📂src
 ┃ ┣ 📂FrameHopper
 ┃ ┃ ┣ 📂util
 ┃ ┃ ┃ ┣ 📜cluster.py
 ┃ ┃ ┃ ┗ 📜obj.py
 ┃ ┃ ┣ 📜agent.py
 ┃ ┃ ┣ 📜environment.py
 ┃ ┃ ┗ 📜run.py
 ┃ ┣ 📂LRLO
 ┃ ┃ ┣ 📂util
 ┃ ┃ ┃ ┣ 📜cal_F1.py
 ┃ ┃ ┃ ┣ 📜cal_quality.py
 ┃ ┃ ┃ ┗ 📜get_state.py
 ┃ ┃ ┣ 📜agent.py
 ┃ ┃ ┣ 📜environment.py
 ┃ ┃ ┗ 📜run.py
 ┃ ┗ 📂Reducto
 ┃ ┃ ┣ 📂util
 ┃ ┃ ┃ ┣ 📂differencer
 ┃ ┃ ┃ ┃ ┣ 📜diff_composer.py
 ┃ ┃ ┃ ┃ ┗ 📜diff_processor.py
 ┃ ┃ ┃ ┣ 📂hashbuilder
 ┃ ┃ ┃ ┃ ┗ 📜hash_builder.py
 ┃ ┃ ┃ ┣ 📜data_loader.py
 ┃ ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┃ ┣ 📜utils.py
 ┃ ┃ ┃ ┗ 📜video_processor.py
 ┃ ┃ ┣ 📜run.py
 ┃ ┃ ┗ 📜simulator.py
 ┣ 📂utils
 ┃ ┗ 📜util.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┗ 📜run.py
 ```