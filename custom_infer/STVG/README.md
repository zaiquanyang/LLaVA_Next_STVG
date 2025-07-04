# 环境配置 ！
 - transformers库中的generate函数默认是 torch.no_grad() 的，如果要返回梯度，一定要注释掉 ！！



# sptial lcoalization

 - test-time contrastive prompt 似乎对静态属性更容易 work, 对于动作的话可能会出错？
 - 在apply test-time contrastive prompt for 空间定位的时候，静态的属性还是要保留对比的两个句子中，不然对比之后的结果可能是指向了错误的物体实例。
 - 至于是只保留要对比的动作，还是把其他不参与比较动作也保留在句子当中，可能需要实验结果去比较

 - 只有当描述的目标物体真实存在某一帧时，针对该帧的对比结果才会更加更新，否则会有噪音

# temporal localization


