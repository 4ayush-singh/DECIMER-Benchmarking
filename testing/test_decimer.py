from DECIMER import predict_SMILES

image_path = "scripts\sample_structure.png"
smiles = predict_SMILES(image_path)

print("Predicted SMILES:", smiles)