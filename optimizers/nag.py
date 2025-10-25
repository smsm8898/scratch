class NAG:
    name = "nag"
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0

    def update(self, w, grad):
        # t 시점에서 w update할 때, 다음 시점에서 사용할 v를 미리 업데이트하여 다음 시점에서 사용
        # 미래에 사용할 lookahead를 미리 계산해놓아서 update를 깔끔하게 할 수 있음
        w = w - self.lr * (grad + self.beta * self.v)
        self.v = self.beta * self.v + grad # lookahead
        return w