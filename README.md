# PandemicAlphaZero
Trying to Play Pandemic with a a simplfied AlphaZero implementation 

<!-- ABOUT THE PROJECT -->
## About The Project

Just a simple terminal connect4 player, ai utilising minimax with alpha/beta pruning and shallow search move ordering. 
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

First install docker, then build the included Dockerfile from within downloaded folder 
  `docker build -t {image_name} -f Dockerfile .`
  
In order to evaluate your trained models make sure to create a volume 
  `docker volume create {volume_name}`
  
Then link your container to the volume when initialising
  `docker run -itd --name {container_name} -v {volume_name}:/learn/checkpoints {image_name}`
  
You can then run commands via
  `docker exec -it {container_name} {command}`

<!-- USAGE EXAMPLES -->
To train a model your command is:
  `python3 learn.py [Args]`

Args are all outlined in the learn.py file itself, some key ones: 
  --n_generations : The number of generations to train for
  --n_ep : The number of episodes to generate per generation
  --model_iter : The id of this run
  --n_processes : How many processes to use when generating episodes
  --external_log : Log game statistics to wandb (requires you to log in with command 'wandb login') 
  
To evaluate a previously trained model your command is:
  `python3 evaluate.py [Args]`
  
Args are all outlined in the evaluate.py file itself, some key ones: 
  --model_iter : The id of this run
  --load_gen : The specific generations of this iteration to evaluate (can pass any number of values)
  --n_ep : The number of episodes to evaluate over
  --n_processes : How many processes to use when generating episodes
  --external_log : Log game statistics to wandb (requires you to log in with command 'wandb login') 

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Thank to the following people for which much of the code of this project was built upon

* Pandemic Model from [PAIndemic](https://github.com/BlopaSc/PAIndemic)
* AlphaZero implementation based upon [AlphaZero General](https://github.com/suragnair/alpha-zero-general)
* With some modificiations based upon [SinglePlayer AlphaZero](https://github.com/tmoer/alphazero_singleplayer)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
