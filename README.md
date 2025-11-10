## Create a virtual environment

- Create a new virtual environment if not existed yet:  
  `python -m venv venv`

- Activate the environment:
  - Windows:  
   `venv\Scripts\activate` 
  - Macs/Linux:  
  `source venv/bin/activate` 

- Install dependencies:  
`pip install -r requirements.txt`


## Scripts
- Download embedding model (ProtTrans):
  ```bash
  python get_embedding_model.py --model_name Rostlab/prot_t5_xl_uniref50 --model_dir ./embedding_model
  ```
  
  Available models:
  - `Rostlab/prot_t5_xl_uniref50` (default, recommended)
  - `Rostlab/prot_t5_xl_bfd`
  - `Rostlab/prot_t5_xxl_uniref50` (larger, more accurate)
  - `Rostlab/prot_t5_xxl_bfd`
  - `Rostlab/prot_bert`
  - `Rostlab/prot_bert_bfd`
