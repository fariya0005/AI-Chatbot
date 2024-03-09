import tkinter as tk
from tkinter import ttk, scrolledtext, INSERT, END, messagebox, filedialog
from datetime import datetime
import random
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import sklearn

pickled_file_path = "pickled_object.pkl"

with open(pickled_file_path, "rb") as file:
    le = pickle.load(file)

file_path1 = "model1.h5"

if file_path1:
            try:
                model = load_model(file_path1)
                print("Model loaded successfully!")
                
            except Exception as e:
                print(f"Error loading the model: {e}")

max_sequence_length = 20

class ChatApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Chat Application")
        self.master.geometry("600x500")
        self.master.resizable(width=True, height=True)
        self.master.configure(bg="#f0f0f0")

        self.chat_frame = tk.Frame(master, bd=5, relief="solid", borderwidth=1, bg="#263238")
        self.chat_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.user_name = "User"
        self.user_profile_label = tk.Label(self.chat_frame, text=f"Welcome, {self.user_name}!", font=("Arial", 12, "italic"), bg="#263238", fg="#ffffff")
        self.user_profile_label.pack(pady=(10, 0))

        self.chat_history = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, state=tk.DISABLED, height=15, width=40, font=("Arial", 12), relief="flat", bd=0, bg="#37474f", fg="#ffffff")
        self.chat_history.pack(expand=True, fill="both", pady=(10, 0), padx=10)

        self.user_input = ttk.Entry(self.chat_frame, width=40, font=("Arial", 12), validate="focusin", validatecommand=self.animate_entry, style="TEntry")
        self.user_input.pack(pady=10, padx=10, ipady=5)

        self.send_button = ttk.Button(self.chat_frame, text="Send", command=self.send_user_message, style="TButton")
        self.send_button.pack(pady=5)

        self.clear_button = tk.Button(self.chat_frame, text="Clear Input", command=self.clear_user_input, font=("Arial", 12), bg="#FF5733", fg="white", relief="flat", bd=0)
        self.clear_button.pack(pady=5)

        self.options_frame = tk.Frame(self.chat_frame, bg="#263238")
        self.options_frame.pack(side=tk.BOTTOM, pady=(0, 10))

        self.history_button = tk.Button(self.options_frame, text="Chat History", command=self.show_chat_history, font=("Arial", 12), bg="#3498db", fg="white", relief="flat", bd=0)
        self.history_button.pack(side=tk.LEFT, padx=10)

        self.delete_chat_button = tk.Button(self.options_frame, text="Delete Chat", command=self.delete_chat, font=("Arial", 12), bg="#e74c3c", fg="white", relief="flat", bd=0)
        self.delete_chat_button.pack(side=tk.RIGHT, padx=10)

        self.save_chat_button = tk.Button(self.options_frame, text="Save Chat", command=self.save_chat, font=("Arial", 12), bg="#2ecc71", fg="white", relief="flat", bd=0)
        self.save_chat_button.pack(side=tk.LEFT, padx=10)

        self.clear_history_button = tk.Button(self.options_frame, text="Clear History", command=self.clear_history, font=("Arial", 12), bg="#e67e22", fg="white", relief="flat", bd=0)
        self.clear_history_button.pack(side=tk.RIGHT, padx=10)

        self.status_bar = tk.Label(master, text="Ready", font=("Arial", 10, "italic"), bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#263238", fg="#ffffff")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.messages = []
        self.emoji_picker_window = None

    def animate_entry(self):
        self.user_input.config(foreground="#263238", background="#ffffff")

    def send_user_message(self):
        user_message = self.user_input.get()
        messages = user_message
        if user_message:
            self.user_input.delete(0, tk.END)
            timestamp = datetime.now().strftime("%H:%M")
            self.messages.append(("You", timestamp, user_message))
            self.slide_in_new_message()
            self.master.after(random.randint(500, 1500), self.send_bot_message)

    def send_bot_message(self, *_):
        from_user = self.user_input.get()
        question_seq = Tokenizer().texts_to_sequences([from_user])
        question_padded = pad_sequences(question_seq, maxlen=max_sequence_length, padding='post')
        resp = model.predict(question_padded)
        x = np.argmax(resp[0])
        lst = np.array([x])
        v = le.inverse_transform(lst)
        
        bot_responses = ["I'm not sure what you mean.", "Interesting!", "Tell me more...", "That's cool!"]
        bot_response = random.choice(bot_responses)
        timestamp = datetime.now().strftime("%H:%M")
        self.messages.append(("Bot", timestamp, v))
        self.slide_in_new_message()

    def slide_in_new_message(self):
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        for sender, timestamp, message in self.messages:
            formatted_message = f"{timestamp} {sender}: {message}\n"
            self.chat_history.insert(tk.END, formatted_message, sender.lower())
            self.apply_message_style(sender.lower())
        self.chat_history.configure(state=tk.DISABLED)
        self.chat_history.see(tk.END)

    def apply_message_style(self, sender):
        if sender == "you":
            self.chat_history.tag_configure(sender, font=("Arial", 12, "bold"), foreground="#ffffff", justify="right")
        else:
            self.chat_history.tag_configure(sender, font=("Arial", 12), foreground="#ffffff", justify="left")

    def clear_user_input(self):
        self.user_input.delete(0, tk.END)

    def show_chat_history(self):
        chat_history_window = tk.Toplevel(self.master)
        chat_history_window.title("Chat History")
        chat_history_window.geometry("400x400")
        history_text = scrolledtext.ScrolledText(chat_history_window, wrap=tk.WORD, state=tk.DISABLED, height=20, width=40, font=("Arial", 12), relief="flat", bd=0, bg="#ffffff", fg="#2e2e2e")
        history_text.pack(expand=True, fill="both", pady=(10, 0), padx=10)

        for sender, timestamp, message in self.messages:
            formatted_message = f"{timestamp} {sender}: {message}\n"
            history_text.insert(tk.END, formatted_message, sender.lower())
            history_text.tag_configure(sender.lower(), font=("Arial", 12), foreground="#2e2e2e", justify="left")

        history_text.configure(state=tk.DISABLED)

    def delete_chat(self):
        confirmed = messagebox.askyesno("Delete Chat", "Are you sure you want to delete the last message?")
        if confirmed and self.messages:
            self.messages.pop()
            self.slide_in_new_message()

    def save_chat(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                for sender, timestamp, message in self.messages:
                    formatted_message = f"{timestamp} {sender}: {message}\n"
                    file.write(formatted_message)
            messagebox.showinfo("Save Chat", "Chat saved successfully!")

    def clear_history(self):
        confirmed = messagebox.askyesno("Clear History", "Are you sure you want to clear the chat history?")
        if confirmed:
            self.messages = []
            self.slide_in_new_message()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)

    app.send_bot_message()  # Initial bot message
    app.send_bot_message("Hello! How can I help you today?")
    app.send_bot_message("This is a simulated conversation.")

    root.mainloop()
