# LLsiM: Large Language Similarity Models

## Recipes Case Base

### Case Composition

The cases of the recipes case base are modeled as NEST workflows. They contain task node, data nodes, and workflow nodes. These nodes are connected via data flow edges, control flow edges, and part-of edges.

There are separate semantic descriptions for the task nodes, data nodes, and workflow nodes. The edges do not have semantic descriptions.

### Similarity Compositon

The semantic description of the workflow nodes consists of the following:

| Attribute              | Type              | Similarity Measure | Weight |
| ---------------------- | ----------------- | ------------------ | ------ |
| name                   | String            | Levenshtein        | 1      |
| source url             | String            | -                  | 0      |
| preparation time (min) | Integer (0, 1000) | Numeric (Linear)   | 1      |
| calories               | Integer (0, 2500) | Numeric (Linear)   | 1      |

The attributes of the workflow nodes are aggregated by the average of the similarities of the attributes.

The semantic description of the task nodes consists of the following:

| Attribute | Type              | Similarity Measure      | Weight |
| --------- | ----------------- | ----------------------- | ------ |
| name      | String (Taxonomy) | Taxonomy (User Weights) | 1      |

The semantic description of the data nodes consists of the following:

| Attribute    | Type                 | Similarity Measure      | Weight |
| ------------ | -------------------- | ----------------------- | ------ |
| name         | String (Taxonomy)    | Taxonomy (User Weights) | 2      |
| amount.value | Integer (0, 1000)    | -                       | 1.1    |
| amount.unit  | String (Enumeration) | Numeric (Linear)        | 1.1    |

The attributes of the task nodes are aggregated by the average of the similarities of the attributes. The attributes of `amount` are nested.

## Cars Case Base

### Case Composition

The cases of the cars case base are modeled as collections of attribute-value pairs (feature vector cases).

### Similarity Composition

The similarity of the cases is calculated by the weighted average of the similarities of the following attribute-value pairs:

| Attribute    | Type                 | Similarity Measure | Weight |
| ------------ | -------------------- | ------------------ | ------ |
| price        | Integer              | Numeric (Linear)   | 1      |
| year         | Integer              | Numeric (Linear)   | 1      |
| manufacturer | String (Taxonomy)    | Taxonomy (Path)    | 1      |
| make         | String               | Levenshtein        | 1      |
| fuel         | String               | Levenshtein        | 1      |
| miles        | Integer (0, 9999999) | Numeric (Linear)   | 1      |
| title_status | String               | Levenshtein        | 1      |
| transmission | String               | Levenshtein        | 1      |
| drive        | String               | Levenshtein        | 1      |
| type         | String               | Levenshtein        | 1      |
| paint_color  | String               | Levenshtein        | 1      |

## Usage

```shell
uv run llsim --help
```

## Cost

- `retrieve-naive`: ca. $4 (two variants, two models)
- `build-preferences-medium`: ca. $5 (two models)
- `build-preferences-small`: ca. $8 (one model)
- `build-similarity`: ca. $0.5 (five models)
