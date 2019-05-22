"""
Adam Tyson | adamltyson@gmail.com | 2019-05-22

"""


import os
import tkinter as tk
from tkinter import messagebox
import tkinter.filedialog
import numpy as np
import options.options_variables as options_var


def gui_run():
    options = get_opt()
    variables = get_var()
    variables['plot_style'] = 'seaborn-muted'
    channel_info = get_im_channels(
        num_coloc_channels=variables['num_coloc_channels'])
    direc = choose_dir()
    return options, variables, direc, channel_info


def get_opt():
    class OptGUI:
        def __init__(self, root):
            self.opt_names,\
                self.opt_prompts,\
                self.opt_defaults =\
                options_var.opt_initialise_all()

            self.opts_get = []
            self.opts = []
            self.option_dict = []
            self.gui_fill()
            self.return_dict()

        def gui_fill(self):
            row = 0
            count = 0
            for opt_prompt in self.opt_prompts:
                opt_tmp = tk.BooleanVar()
                opt_tmp.set(self.opt_defaults[count])
                self.opts_get.append(opt_tmp)
                tk.Label(root, text=opt_prompt, height=2).grid(
                    row=row, sticky=tk.W, columnspan=2)
                tk.Radiobutton(root, text="Yes",
                               variable=opt_tmp, value=True).grid(
                    row=row+1, sticky=tk.W, column=0)
                tk.Radiobutton(root, text="No", variable=opt_tmp,
                               value=False).grid(
                    row=row+1, sticky=tk.W, column=1)

                row = row+2
                count = count + 1

            tk.Button(root, text="Proceed",
                      command=self.quit_loop).grid(row=row, column=2)

            # <Return> to progress
            root.bind('<Return>', lambda event: self.quit_loop())
            # to bring to front
            root.attributes("-topmost", True)
            root.focus_force()
            root.mainloop()

        def quit_loop(self):
            for var in self.opts_get:
                self.opts.append(var.get())

            root.destroy()

        def return_dict(self):
            self.option_dict = dict(zip(self.opt_names, self.opts))

    root = tk.Tk()
    root.title('Options')
    opt = OptGUI(root)
    return opt.option_dict


def get_var():
    class OptGUI:
        def __init__(self, root):
            self.var_names, self.var_prompts, self.var_defaults =\
                options_var.var_initialise_all()

            self.var_get = []
            self.var = []
            self.var_float = []
            self.variable_dict = []
            self.gui_fill()
            self.return_dict()

        def gui_fill(self):
            row = 0
            count = 0
            for var_prompt in self.var_prompts:
                var_tmp = tk.StringVar()
                self.var_get.append(var_tmp)

                label_text = tk.StringVar()
                label_text.set(var_prompt)
                label = tk.Label(root, textvariable=label_text, height=4)
                label.grid(row=row)

                text = tk.Entry(root, textvariable=var_tmp)
                text.insert(tk.END, self.var_defaults[count])  # default
                text.grid(row=row, column=1)

                row = row+1
                count = count + 1

            tk.Button(root, text="Proceed",
                      command=self.quit_loop).grid(row=row, column=2)

            # <Return> to progress
            root.bind('<Return>', lambda event: self.quit_loop())

            # to bring to front
            root.attributes("-topmost", True)
            root.focus_force()
            root.mainloop()

        def quit_loop(self):
            for variable in self.var_get:
                self.var.append(variable.get())
            root.destroy()

        def return_dict(self):
            self.var_float = list(np.float_(self.var))
            self.variable_dict = dict(zip(self.var_names, self.var_float))

    while True:
        try:
            root = tk.Tk()
            root.title('Variables')
            var = OptGUI(root)
        except ValueError:
            error = tk.Tk()
            error.withdraw()
            messagebox.showinfo(
                'Input error', 'Please try again, enter numerical values')
            error.destroy()
            continue
        break

    variable_dict = var.variable_dict
    variable_dict = options_var.var_force(variable_dict)

    return variable_dict


def choose_dir():
    # choose a directory and move into it
    root = tk.Tk()
    root.withdraw()
    im_dir = tk.filedialog.askdirectory(title='Select image directory')
    root.update() # otherwise doesn't close on OSX w PyCharm
    os.chdir(im_dir)
    root.destroy()
    return im_dir


def get_plot_var(files):
    class OptGUI:
        def __init__(self, root, files):

            self.var_names = files
            # self.var_prompts = files
            self.var_get_group = []
            self.var_get_refframe = []

            self.var_group = []
            self.var_list_group = []
            self.variable_dict_group = []
            self.var_refframe = []
            self.var_int_refframe = []
            self.variable_dict_refframe = []
            self.gui_fill()
            self.return_dict()

        def gui_fill(self):
            label_text = tk.StringVar()
            label_text.set('Image')
            label = tk.Label(root, textvariable=label_text, height=4)
            label.grid(row=0)

            label_text = tk.StringVar()
            label_text.set('Group')
            label = tk.Label(root, textvariable=label_text, height=4)
            label.grid(row=0, column=1)

            label_text = tk.StringVar()
            label_text.set('Reference frame (0 = 1st)')
            label = tk.Label(root, textvariable=label_text, height=4)
            label.grid(row=0, column=2)
            row = 1
            count = 0
            for var_name in self.var_names:
                var_tmp_group = tk.StringVar()
                var_tmp_refframe = tk.StringVar()

                self.var_get_group.append(var_tmp_group)
                self.var_get_refframe.append(var_tmp_refframe)

                label_text = tk.StringVar()
                label_text.set(var_name)
                label = tk.Label(root, textvariable=label_text, height=4)
                label.grid(row=row)

                text_group = tk.Entry(root, textvariable=var_tmp_group)
                text_group.insert(tk.END, '0')  # default
                text_group.grid(row=row, column=1)

                text_refframe = tk.Entry(root, textvariable=var_tmp_refframe)
                text_refframe.insert(tk.END, 0)  # default
                text_refframe.grid(row=row, column=2)

                row = row+1
                count = count + 1

            tk.Button(root, text="Proceed",
                      command=self.quit_loop).grid(row=row, column=2)

            # <Return> to progress
            root.bind('<Return>', lambda event: self.quit_loop())

            # to bring to front
            root.attributes("-topmost", True)
            root.focus_force()
            root.mainloop()

        def quit_loop(self):
            for variable in self.var_get_group:
                self.var_group.append(variable.get())
            for variable in self.var_get_refframe:
                self.var_refframe.append(variable.get())
            root.destroy()

        def return_dict(self):
            self.var_list_group = list(self.var_group)
            self.variable_dict_group =\
                dict(zip(self.var_names, self.var_list_group))
            self.var_int_refframe = list(np.int_(self.var_refframe))
            self.variable_dict_refframe =\
                dict(zip(self.var_names, self.var_int_refframe))

    root = tk.Tk()
    root.title('Choose plotting options')
    var = OptGUI(root, files)

    variable_dict_group = var.variable_dict_group
    variable_dict_refframe = var.variable_dict_refframe

    return variable_dict_group, variable_dict_refframe


def get_im_channels(num_coloc_channels=1):
    class ChannelNameGUI:
        def __init__(self, root, num_coloc_channels=1):
            self.channel_names, self.channel_prompts, self.channel_defaults =\
                options_var.channel_initialise(
                    num_coloc_channels=num_coloc_channels)

            self.var_get = []
            self.var = []
            self.var_float = []
            self.channels_dict = []
            self.gui_fill()
            self.return_dict()

        def gui_fill(self):
            row = 0
            count = 0
            for prompt in self.channel_prompts:
                var_tmp = tk.StringVar()
                self.var_get.append(var_tmp)

                label_text = tk.StringVar()
                label_text.set(prompt)
                label = tk.Label(root, textvariable=label_text, height=4)
                label.grid(row=row)

                text = tk.Entry(root, textvariable=var_tmp)
                text.insert(tk.END, self.channel_defaults[count])  # default
                text.grid(row=row, column=1)

                row = row+1
                count = count + 1

            tk.Button(root, text="Proceed",
                      command=self.quit_loop).grid(row=row, column=2)

            # <Return> to progress
            root.bind('<Return>', lambda event: self.quit_loop())

            # to bring to front
            root.attributes("-topmost", True)
            root.focus_force()
            root.mainloop()

        def quit_loop(self):
            for variable in self.var_get:
                self.var.append(variable.get())
            root.destroy()

        def return_dict(self):
            self.channels_dict = dict(zip(self.channel_names, self.var))

    root = tk.Tk()
    root.title('Choose channels')
    var = ChannelNameGUI(root, num_coloc_channels=num_coloc_channels)
    channels_dict = var.channels_dict

    return channels_dict
