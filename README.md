# ProtoCRL_SAC

An RL agent (SAC + GMM for task-clustering) tested on 4 ContinualWorld (Meta-World v2) tasks. This repository includes a custom mean routing mechanism based on negative log-likelihood to prevent GMM collapse.

**Note:** This is not a polished library. It was built to test and understand how ProtoCRL works in an high-dimensional environment with continuous actions.

## Full Writeup
The complete methodology, debugging process and analysis of the results can be found in the attached report: 
**[PUT LINK LATER OF THE PDF]**

## Findings
* **The Failure Mode:** Using the GMM's ELBO loss to train the encoder caused severe representation collapse, the standard posterior collapsed onto a single cluster from initialization, because unused components never received usable gradient once their assignment probability approached zero. Moreover using ReLU on the last encoder's layer brought its std to drop to 0.
* **The Fix:** Decoupling the encoder from the GMM and implementing a negative log-likelihood novelty trigger allowed the system to dynamically route tasks. The encoder's output is moved into an hypersphere.
* **The Result:** The routing mechanism successfully reduced catastrophic forgetting compared to a vanilla SAC baseline **ADD A BRIEF RECAP OF THE RESULTS OF THE RUN COMPARED TO SAC**

## Running the Code

This project requires a working MuJoCo 200 setup. You can install the exact dependencies using the provided requirements file.

**Note:** The top of the script include an AI-generated workaround block for known mujoco-py numpy bugs and Meta-World V1/V2 repacking issues. 
