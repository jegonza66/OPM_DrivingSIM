# OPM DrivingSim Project

## An OPM / Eye Tracker Coregistration Study

### Project Structure

To run the scripts from this project please set the directory as follows:

```bash
├───DATA
│   ├───OPM
│   │   ├───11766
│   │   └───...
│   │               
│   ├───ET_DATA
│   │   ├───11766    
│   │   └───...
│   │              
│   └───Digitisation
│       ├───11766    
│       └───...
│                              
└───Scripts
    ├───connectivity.py
    ├───paths.py
    ├───setup.py
    └───...
```

Brief explanation of the main modules:

- paths.py: Module including paths to data and save directories.
- setup.py: Module defining:
    - Experiment information in the exp_info class, that contains information about every subjects scanning.
    - Subject object in raw_subject class, that will be used in the preprocessing module and will store information about the subjects preprocessing, scanning and
      behabioural and Eye-tracker data.
- plot_general.py: Module with plotting functions for main analysis.
- save.py and load.py: Modules to save and load variables, figures, and objects (preprocessed subjects and MEG data).
- connectivity.py: Module with functions to calculate connectivity matrices.