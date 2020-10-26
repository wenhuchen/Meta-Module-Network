wget https://gqadata.s3-us-west-2.amazonaws.com/questions.zip
unzip questions.zip
wget https://gqadata.s3-us-west-2.amazonaws.com/scene_graph.zip
unzip scene_graph.zip
mkdir sceneGraphs
mv trainval_bounding_box.json sceneGraphs
wget https://convaisharables.blob.core.windows.net/meta-module-network/gqa_visual_features.zip
unzip gqa_visual_features.zip
