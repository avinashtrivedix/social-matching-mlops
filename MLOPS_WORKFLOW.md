# MLOps Standard Operating procedure
## The Daily Workflow
1. Write code and train the model.
2. Track any new heavy files with DVC: 'dvc add path/to/heavy_file.csv'
3. Stage all code and DVC reciepts for Git: 'git add .'
4. COmmit the snapshot : 'git commit -m "Breif description of changes"'
5. Push the Code to github : 'git push'