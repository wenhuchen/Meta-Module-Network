
## Meta-Function Definition

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

## Meta Module Network

All the training data (under the questions/ folder) given to the networks are called '*_inputs.json', these files are simply a restructured data format (containing the dependency between the execution from different steps) from the original "*_programs.json" files. 

- *_programs.json
```
    "2354786",
    "Is the sky dark?",
    [
      "[2486325]=select(sky)",
      "?=verify([0], dark)"
    ],
    "02930152",
    "yes"
```
- *_inputs.json
```
    "2354786",
    "Is the sky dark?",
    [],
    [
      [
        "select",
        null,
        null,
        null,
        "sky",
        null,
        null,
        null
      ],
      [
        "verify",
        null,
        "dark",
        null,
        null,
        null,
        null,
        null
      ]
    ],
    [
      [], 
      [
        [
          1,
          0
        ]
      ]
    ],
    "02930152",
    "yes"
```

In the input file, the following data type is called program recipe, corresponding to "[2486325]=select(sky)".
```
[
  "select",
  null,
  null,
  null,
  "sky",
  null,
  null,
  null
],
```
In the input file, the following data type is called layer dependency. In the 0-th element (1st layer of MMN), there is only a [], which means nothing is dependent on the previous layer. In the 1-th element (2nd layer of MMN), there is a [1, 0], which means that the 1-st node's is dependent on 0-th node's output (e.g. "?=verify([0], dark)") in this layer. 
```
    [
      [], 
      [
        [
          1,
          0
        ]
      ]
    ],
```
The dependency relationship is the critical part in MMN, for example, the dependency of "[[], [[1,0], [3,0]], [[2,1]], [[4,2], [4,3]]]" is visualized as below: 
<p>
<img src="introduction.png" width="800">
</p>
