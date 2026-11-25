# 代码样例

## 参考解答如何改名

训练题 `train:0` 中，第一月卖出 48 个发夹，第二月卖出第一月的一半，求两个月的总数。下面是同一条训练样本在两个数据版本中的参考程序，两者均返回 72。

**实验 8 · style**

```python
def simple_math_problem() -> int:
    clips_april = 48
    clips_may = clips_april / 2
    total_clips = clips_april + clips_may
    result = (int(total_clips))
    return result
```

**实验 12 · vars**

```python
def simple_math_problem() -> int:
    x_1 = 48
    x_2 = x_1 / 2
    x_3 = x_1 + x_2
    result = (int(x_3))
    return result
```

两个版本保留 `result` 及整个计算过程。使用相同 tokenizer 后，这条参考程序从 57 个代码 token 变为 54 个。

## 模型实际生成的代码

验证题 `train:780` 来自 GSM8K 官方训练划分，在本项目中划入验证集。题意是每小时读 12 页，阅读速度提高到三倍，两小时能读几页。标准答案为 72。下面保留了两次实验在 T=1、32 步去噪时的模型输出。

**实验 8 · 执行失败**

```python
def simple_math_problem() -> int:
    pages_per_per = 12
    pages_per_2h = (2 * pages_per_hour)/3
    total_pages = pages_per_2h * 2
    result = (total_pages)
    return result
```

程序把初始速度存为 `pages_per_per`，随后引用了未定义的 `pages_per_hour`。实际执行记录为 `NameError: name 'pages_per_hour' is not defined`。计算式也使用了除以 3，即使修正变量名，仍然得不到正确结果。

**实验 12 · 答对**

```python
def simple_math_problem() -> int:
    x_1 = 12
    x_2 = 3
    x_3 = 2
    x_4 = (x_1 * x_2) * x_3
    result = (x_4)
    return result
```

程序将阅读速度乘以 3 再乘以 2 小时，返回 72。这组样例展示了单题的变化。639 道验证题中有 100 题由错变对，27 题由对变错，21 题两次均正确。
