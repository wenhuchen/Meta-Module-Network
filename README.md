# Meta-Module-Network
Code for Paper "Meta Module Network for Compositional Visual Reasoning"

This repository contains partial components, the full version will be released in the future. The released components are listed below:
1. The generated programs from the program generator.
2. The symbolic execution for the visual question answering.

## Function Definition
We define roughly 20+ functions based on the semantic-str provided in the original GQA dataset and categorize them into the following classes:
1. relate: finding objects with name and relation constraint, two version: normal + reverse.
2. filter: filter objects with given attributes or filter objects based on horizontal or vertical geometric position. 
3. filter_not: filter objects without given attributes
4. query: query object names or attributes or positions in the scene graph
5. verify: verify whether the objects contain certain attributes or their horizontal/vertial positions.
6. verify_rel: verify relations between different objects in the image.
7. choose: choose which attributes or names or geometric locations the current object has.
8. choose_rel: choose which relation the objects have.
9. and/or/exist: logical operations.
10: different: whether the objects are different
11: same: whether the objects are the same.
12: commmon: what attributes the objects have in common.
13: same_attr: whether the objects have the same given attributes.
14: different_attr: whether the objects have different given attributes.

## Description of different files
- API_provider.py: define the functions inside the API
- sceneGraphs/trainval_bounding_box.json: the scene graph provided by the original GQA dataset
  ```
    {
      imageId:
      {
        bouding_box_id:
        {
          x: number,
          y: number,
          w: number,
          h: number,
          relations: [{object: "bounding_box_id", name: "relation_name"} ... ],
          name: object_class,
          attributes: [attr1, attr2, ... ]
        },
        bouding_box_id:
        {
          ...
        },
      }
    }
  ```
- GQA_hypernym.py: define the hypernym/hyponym we used in string matching during execution
- questions: the questions-program pairs and their associated images.
  ```
  [
    [
      "ImageId",
      "Question",
      "Programs": [f1, f2, ..., fn],
      "QuestionId",
      "Answer"
    ]
  ]
  ```
  
## Symbolic Execution
We can run the run.py to perform symbolic execution on the GQA provided scene graph to get the answer.
