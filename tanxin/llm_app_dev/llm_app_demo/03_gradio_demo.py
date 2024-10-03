# -*- coding: utf-8 -*-

'''
gradio demo

'''
import gradio as gr
import random

def greet(name, intensity):
    return 'hello, ' + name + '!' * int(intensity)


def demo1Fun():

    '''
    tutorial中的示例
    gr.Interface :  使用
    '''

    demo = gr.Interface(fn=greet, inputs=["text", "slider"], outputs=["text"])
    demo.launch()


def randomResponse(message, history):
    return random.choice(["yes", "no"])

def demo2Fun():
    
    '''
    gr.ChatInterface : 对话机器人
    '''
    demo2 = gr.ChatInterface(fn=randomResponse)
    demo2.launch()


if __name__ == '__main__':
    demo2Fun()
