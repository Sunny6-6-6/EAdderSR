import torch
# from basicsr.models.archs.srresnet_arch import MSRResNet, MSRResNet_Adder

# from vdsr_adder import Net as ann
# from origin_adder import adder2d

mse = torch.nn.MSELoss()
cross_e = torch.nn.CrossEntropyLoss()


class Transform_final(torch.nn.Module):

    def __init__(self):
        super(Transform_final, self).__init__()
        self.linear_trans = torch.nn.Conv2d(3, 1, kernel_size=5, stride=5, bias=False)
        self.linear_trans.weight = torch.nn.Parameter(torch.ones(1,3,5,5))
        # self.pa = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        # x =  ((x - x.reshape(x.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1)) / (
        #         #             x.reshape(x.shape[0], -1).max(dim=1).values.reshape(-1, 1, 1, 1) - x.reshape(x.shape[0], -1).min(
        #         #         dim=1).values.reshape(-1, 1, 1, 1)))
        self.x = self.linear_trans(x)
        return x


class Transform_middle(torch.nn.Module):

    def __init__(self):
        super(Transform_middle, self).__init__()
        self.linear_trans = torch.nn.Conv2d(64, 1, kernel_size=5, stride=5, bias=False)
        self.linear_trans.weight = torch.nn.Parameter(torch.ones(1, 64, 5, 5))
        # self.bn = torch.nn.BatchNorm2d(64)
        # self.pc = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        # x = self.bn(x)
        x =  ((x - x.reshape(x.shape[0], -1).mean(dim=1).reshape(-1, 1, 1, 1)) / (
                    x.reshape(x.shape[0], -1).max(dim=1).values.reshape(-1, 1, 1, 1) - x.reshape(x.shape[0], -1).min(
                dim=1).values.reshape(-1, 1, 1, 1)))
        x = self.linear_trans(x)

        return x


class KD_Hooks:
    def __init__(self, is_cuda):
        self.teacher_o_lists = {}
        self.student_o_lists = {}
        self.is_cuda = is_cuda

    def teacher_hook(self, module, feature_in, feature_output):
        self.teacher_o_lists[0] = feature_output

    def student_hook(self, module, feature_in, feature_output):
        self.student_o_lists[0] = feature_output

    def reset(self):
        self.teacher_o_lists = {}
        self.student_o_lists = {}



class P_KD(torch.nn.Module):

    def __init__(self,teacher,student,is_cuda=True):
        super(P_KD, self).__init__()
        self.hooks = KD_Hooks(is_cuda)
        self.is_cuda = is_cuda

        self.trans_final = Transform_final().eval()
        self.trans_middle = Transform_middle().eval()

#         for i, m in enumerate(teacher.body):
#             m.register_forward_hook(hook=self.hooks.teacher_hook)
        teacher.conv_last.register_forward_hook(hook=self.hooks.teacher_hook)
            # break

#         for i, m in enumerate(student.body):
#             m.register_forward_hook(hook=self.hooks.student_hook)
        student.conv_last.register_forward_hook(hook=self.hooks.student_hook)
            # break

    def forward(self):
        if self.is_cuda:
            l_mid = torch.zeros(1).cuda()
            l_blend = torch.zeros(1).cuda()
        else:
            l_mid = torch.zeros(1)
            l_blend = torch.zeros(1)

        for key,value in self.hooks.teacher_o_lists.items():
            t = value
            s = self.hooks.student_o_lists[key]

            if key > 0:
                pass
#                 t_o = t
#                 s_o = s
#                 # print('t',t_o)
#                 # print('s',s_o)
#                 s_o = self.trans_middle(s_o)
#                 t_o = self.trans_middle(t_o)

#                 l_mid = l_mid + torch.nn.L1Loss()(t_o, s_o)

            elif key==0:

                s = self.trans_final(s)
                t = self.trans_final(t)
                l_mid = l_mid + torch.nn.L1Loss()(s, t)
                # print(l_mid)

                    # break
        self.hooks.reset()

        return l_mid


# def P_KD(input, target, teacher, student, is_cuda=True):
#     hooks = KD_Hooks(is_cuda)
#
#     for i, m in enumerate(teacher.body):
#         m.register_forward_hook(hook=hooks.teacher_hook)
#     for i, m in enumerate(student.body):
#         m.register_forward_hook(hook=hooks.student_hook)
#
#     t_o = teacher(input)
#     s_o = student(input)
#
#     if is_cuda:
#         l_mid = torch.zeros(1).cuda()
#         l_blend = torch.zeros(1).cuda()
#     else:
#         l_mid = torch.zeros(1)
#         l_blend = torch.zeros(1)
#
#     for s, t in zip(hooks.student_o_lists, hooks.teacher_o_lists):
#         l_mid = l_mid + mse(s, t)
#
#     # print(hooks.KD_loss())
#     return l_mid
#
#
# input = torch.randn(1, 3, 20, 20).cuda()
# teacher = MSRResNet().cuda()
# student = MSRResNet_Adder().cuda()
#
# P_KD(input, 2, teacher, student, is_cuda=True)