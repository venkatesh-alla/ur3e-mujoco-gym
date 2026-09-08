from typing import Any                                                                                                           
                                                                                                                                 
import numpy as np                                                                                                               
import optuna                                                                                                                    
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise                                       
from torch import nn as nn                                                                                                       
                                                                                                                                 
from rl_zoo3 import linear_schedule                                                                                              
                                                                                                                                 
                                                                                                                                 
def sample_ppo_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:                
    """                                                                                                                          
    Sampler for PPO hyperparams.                                                                                                 
                                                                                                                                 
    :param trial:                                                                                                                
    :return:                                                                                                                     
    """                                                                                                                          
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])                                         
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512])                   
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])                                    
    #learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    learning_rate = 0.001                                                      
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)                                                        
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])                                
    n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])                                                             
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.92, 0.95, 0.98, 0.99, 1.0])                                
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["currently_used"])                                           
    # Uncomment for gSDE (continuous actions)                                                                                    
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)                                                                  
    # Uncomment for gSDE (continuous action)                                                                                     
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])                              
    # Orthogonal initialization                                                                                                  
    ortho_init = False                                                                                                           
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])                                                        
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])                          
    activation_fn_name = trial.suggest_categorical("activation_fn", ["relu"])                                            
    # lr_schedule = "constant"                                                                                                   
    # Uncomment to enable learning rate schedule                                                                                 
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])                                             
    # if lr_schedule == "linear":                                                                                                
    #     learning_rate = linear_schedule(learning_rate)                                                                         
                                                                                                                              
    # TODO: account when using multiple envs                                                                                  
    if batch_size > n_steps:                                                                                                  
        batch_size = n_steps                                                                                                  
                                                                                                                              
    # Independent networks usually work best                                                                                  
    # when not working with images                                                                                            
    net_arch = {                                                                                                              
        "tiny": dict(pi=[64], vf=[64]),                                                                                       
        "small": dict(pi=[64, 64], vf=[64, 64]),                                                                              
        "medium": dict(pi=[256, 256], vf=[256, 256]),
        "currently_used": dict(pi=[512, 512], vf=[512, 512]),                                                                         
    }[net_arch_type]                                                                                                          
                                                                                                                              
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn_name]         
                                                                                                                              
    return {                                                                                                                  
        "n_steps": n_steps,                                                                                                   
        "batch_size": batch_size,                                                                                             
        "gamma": gamma,                                                                                                       
        "learning_rate": learning_rate,                                                                                       
        "ent_coef": ent_coef,                                                                                                 
        "clip_range": clip_range,                                                                                             
        "n_epochs": n_epochs,                                                                                                 
        "gae_lambda": gae_lambda,                                                                                             
        "max_grad_norm": max_grad_norm,                                                                                       
        "vf_coef": vf_coef,                                                                                                   
        # "sde_sample_freq": sde_sample_freq,                                                                                 
        "policy_kwargs": dict(                                                                                                
            # log_std_init=log_std_init,                                                                                      
            net_arch=net_arch,                                                                                                
            activation_fn=activation_fn,                                                                                      
            ortho_init=ortho_init,                                                                                            
        ),                                                                                                                    
    }                                                                                                                         

HYPERPARAMS_SAMPLER = {
    "ppo": sample_ppo_params
}
