# GraphWater-Net
 For exploring the influence of water molecules on predicting protein-ligand binding affinity, we developed a model called GraphWater-Net for predicting protein-ligand binding affinity, which used topological structures to represent protein atoms, ligand atoms, water molecules, and their interactions. Using Graphormer network to extract the interaction features between nodes and other nodes in the topology, as well as the interaction features of edges and nodes, GraphWater-Net generated embeddings with attention weights, and finally input them into softmax for regression prediction to output the predicted binding affinity value. This study demonstrated superior performance on the Comparative Assessment of Scoring Functions (CASF) 2016 testset. 
# Related Files
model.py->model construction
train.py->train model
dataset.py->process dataset
data->PDBbind data and the list of training data and test data
AEScore, CGraphDTA, DataDTA, ECIFGraph HM-Holo-Apo, GraphscoreDTA, SIGN, HACNet->comparison methods
