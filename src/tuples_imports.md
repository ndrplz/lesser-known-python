# Tuples can be useful for organizing your imports 

You start a new project and you open your first blank file, and you feel that omnipotence 
sensation given by the infinite possibilities that this new project promises to unleash. Still, chances are that your
first keystrokes will be devoted to something quite down-to-earth. Like an `import` statement.

For example, 
```python
import os
```
or maybe 
```python
import numpy as np
```
Does that sound familiar? Cool! No omnipotence sensation you say? Let's move on anyway.

After coding for a while, you may (should?) start to feel the need to split your code into different modules.
So then you start to add a couple of innocent import statemets, like:
```python
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
```

Nothing wrong with that. The thing is, as soon as your project grows, the number of imports you do is likely to grow as well.
So you'll end up in the need of importing *a lot* of things:

```python
from tensorflow.python.ops import array_ops, control_flow_ops, data_flow_grad, data_flow_ops, io_ops, math_ops, random_ops, sparse_grad, variable_scope
```
Now, you see the import line has become a bit long. According to which convention you follow, you may want to split it 
into two or more pieces. Possibly, you may want one import each line:

```python
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_grad
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope
```
While I personally don't mind this solution, I agree that it is redundant and verbose.

So why not simply breaking the line?
```python
from tensorflow.python.ops import array_ops, control_flow_ops, data_flow_grad, data_flow_ops, \
    io_ops, math_ops, random_ops, sparse_grad, variable_scope
```
Well, that's just really ugly, you know. So where tuples come into play? Well, here it goes:
```python
from tensorflow.python.ops import (array_ops, control_flow_ops, data_flow_grad, data_flow_ops,
                                   io_ops, math_ops, random_ops, sparse_grad, variable_scope)
```
Isn't that much nicer?
