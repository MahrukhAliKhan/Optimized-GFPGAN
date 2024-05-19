# Optimized GFPGAN Model

This repository provides optimized version of GFPGAN model. The optimized model will take less inference time and resources consumption for processing input images. 

Following are the steps to use this model

#### Cloning the Main Repository :
```
git clone  https://github.com/MahrukhAliKhan/Optimized-GFPGAN.git
```

#### Navigate to Main Repository :
```
cd Optimized-GFPGAN
```

#### Creating Virtual Enviromrnt :
```
sudo apt install python3.9-venv
python3.9 -m venv venv
source venv/bin/activate
```
#### Installing requirements and dependencies:
```
pip install -r requirements.txt
```
#### Running API Code:
```
uvicorn fastapi:app --host 0.0.0.0 
```