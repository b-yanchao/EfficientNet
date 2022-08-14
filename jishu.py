import tkinter


top = tkinter.Tk()
top.title("车辆计数")
top.geometry("300x260")
def xkcUp(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = xkc
    xkc = a+1
    lb_xkc.config(text=xkc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def dkcUp(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = dkc
    dkc = a+1
    lb_dkc.config(text=dkc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def xhcUp(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = xhc
    xhc = a+1
    lb_xhc.config(text=xhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def zhcUp(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc, count
    a = zhc
    zhc = a+1
    lb_zhc.config(text=zhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def jzxcUp(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = jzxc
    jzxc = a+1
    lb_jzxc.config(text=jzxc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def dhcUp(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = dhc
    dhc = a+1
    lb_dhc.config(text=dhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def xkcDown(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = xkc
    if(xkc!=0):
        xkc = a-1
    lb_xkc.config(text=xkc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def dkcDown(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = dkc
    if (dkc != 0):
        dkc = a-1
    lb_dkc.config(text=dkc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def xhcDown(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = xhc
    if (xhc != 0):
        xhc = a-1
    lb_xhc.config(text=xhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def zhcDown(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = zhc
    if (zhc != 0):
        zhc = a-1
    lb_zhc.config(text=zhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def jzxcDown(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = jzxc
    if (jzxc != 0):
        jzxc = a-1
    lb_jzxc.config(text=jzxc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def dhcDown(a):
    global xkc, dkc, xhc, zhc, jzxc, dhc,count
    a = dhc
    if (dhc != 0):
        dhc = a-1
    lb_dhc.config(text=dhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)
def clear():
    global xkc, dkc, xhc, zhc, jzxc, dhc, count
    xkc = 0
    dkc = 0
    xhc = 0
    zhc = 0
    jzxc = 0
    dhc = 0
    lb_xkc.config(text=xkc)
    lb_dkc.config(text=dkc)
    lb_xhc.config(text=xhc)
    lb_zhc.config(text=zhc)
    lb_jzxc.config(text=jzxc)
    lb_dhc.config(text=dhc)
    count = xkc + dkc + xhc + zhc + jzxc + dhc
    sum.config(text=count)

xkc = 0
dkc = 0
xhc = 0
zhc = 0
jzxc = 0
dhc = 0
count = 0

lb_xkc = tkinter.Label(top, text=xkc)
lb_dkc = tkinter.Label(top, text=dkc)
lb_xhc = tkinter.Label(top, text=xhc)
lb_zhc = tkinter.Label(top, text=zhc)
lb_jzxc = tkinter.Label(top, text=jzxc)
lb_dhc = tkinter.Label(top, text=dhc)

Text_xkc = tkinter.Label(top, text="小客车")
Text_dkc = tkinter.Label(top, text="大客车")
Text_xhc = tkinter.Label(top, text="小货车")
Text_zhc = tkinter.Label(top, text="中货车")
Text_jzxc = tkinter.Label(top, text="集装箱车")
Text_dhc = tkinter.Label(top, text="大货车")

clear = tkinter.Button(top, text="重置", command=clear)
sum_t = tkinter.Label(top, text="总数:")
sum = tkinter.Label(top, text=count)


xkcB = tkinter.Button(top, text="增加", command=lambda:xkcUp(xkc))
dkcB = tkinter.Button(top, text="增加", command=lambda:dkcUp(dkc))
xhcB = tkinter.Button(top, text="增加", command=lambda:xhcUp(xhc))
zhcB = tkinter.Button(top, text="增加", command=lambda:zhcUp(zhc))
jzxcB = tkinter.Button(top, text="增加", command=lambda:jzxcUp(jzxc))
dhcB = tkinter.Button(top, text="增加", command=lambda:dhcUp(dhc))

xkcD = tkinter.Button(top, text="减少", command=lambda:xkcDown(xkc))
dkcD = tkinter.Button(top, text="减少", command=lambda:dkcDown(dkc))
xhcD = tkinter.Button(top, text="减少", command=lambda:xhcDown(xhc))
zhcD = tkinter.Button(top, text="减少", command=lambda:zhcDown(zhc))
jzxcD = tkinter.Button(top, text="减少", command=lambda:jzxcDown(jzxc))
dhcD = tkinter.Button(top, text="减少", command=lambda:dhcDown(dhc))

clear.place(x=0,y=0)
sum_t.place(x=220,y=0)
sum.place(x=255,y=0)

Text_xkc.place(x=20,y=25)
lb_xkc.place(x=80,y=25)
xkcB.place(x=140,y=20)
xkcD.place(x=220,y=20)

Text_dkc.place(x=20,y=65)
lb_dkc.place(x=80,y=65)
dkcB.place(x=140,y=60)
dkcD.place(x=220,y=60)

Text_xhc.place(x=20,y=105)
lb_xhc.place(x=80,y=105)
xhcB.place(x=140,y=100)
xhcD.place(x=220,y=100)

Text_zhc.place(x=20,y=145)
lb_zhc.place(x=80,y=145)
zhcB.place(x=140,y=140)
zhcD.place(x=220,y=140)

Text_jzxc.place(x=20,y=185)
lb_jzxc.place(x=80,y=185)
jzxcB.place(x=140,y=180)
jzxcD.place(x=220,y=180)

Text_dhc.place(x=20,y=225)
lb_dhc.place(x=80,y=225)
dhcB.place(x=140,y=220)
dhcD.place(x=220,y=220)


top.mainloop()