# DECIMER-Benchmarking
Conda python environment repository for benchmarking DECIMER for molecule to SMILES conversion from research paper pdfs.

Requirements:

Conda or Miniconda.

Python 3.10.

DECIMER Image Transformer.

DECIMER-Segmentation.

Poppler for PDF processing, as required by the segmentation workflow.

Sufficient RAM and disk space for processing PDF files and neural-network models.

An internet connection during initial installation and model downloads.

A GPU is optional for many workflows, although compatible hardware and dependencies can improve processing speed.

1. Clone the repository

   git clone https://github.com/4ayush-singh/DECIMER-Benchmarking/

   cd decimer-pdf-smiles

3. Create and activate the Conda environment

   conda create -n decimer-pdf python=3.10 pip -y

   conda activate decimer-pdf

5. Install system dependencies

   Install Poppler using Conda:

   conda install -c conda-forge poppler -y

7. Install Python dependencies

   python -m pip install -r requirements.txt

9. Verify the installation

   Verify that DECIMER can be imported:

   python -c "from DECIMER import predict_SMILES; print('DECIMER import successful')"
   
   Verify RDKit:

   python -c "from rdkit import Chem; print('RDKit import successful')"
   
   Check the installed package versions:

   python -m pip show decimer decimer-segmentation rdkit

   If all checks succeed, the environment is ready for testing the PDF-processing workflow.

11. Execute run_extraction.py file and evaluate_smiles.py files.

NOTE:

requirements.txt: the project's intended Python dependencies.

requirements-lock.txt: the installed Python package versions captured after testing.

environment.yml: the Conda environment specification, including Python and Conda-managed dependencies.

Run test_decimer.py file in testing folder to check if Decimer's running or not.
