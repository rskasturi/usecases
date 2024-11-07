# Distributed Data Parallel

References:

[Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

[Multi-GPU AI Training (Data-Parallel) with Intel® Extension for PyTorch](https://www.youtube.com/watch?v=3A8AVsNNHOg)

## Run Commands

To run the code, run all instances since the program waits for each world size to be available before starting execution:
```bash
python ddp_demo.py --world-size X --rank Y --xpu Z
```
where,

> X is the number of instances

> Y is the rank of each instances running in Z device

> Z is the device number

For example, to run  16 instances, 2 ranks on each device,
```bash
python ddp_demo.py --world-size 16 --rank 0 --xpu 0
python ddp_demo.py --world-size 16 --rank 1 --xpu 0
python ddp_demo.py --world-size 16 --rank 2 --xpu 1
python ddp_demo.py --world-size 16 --rank 3 --xpu 1
..
python ddp_demo.py --world-size 16 --rank 14 --xpu 7
python ddp_demo.py --world-size 16 --rank 15 --xpu 7
```