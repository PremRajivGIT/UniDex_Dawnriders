import os
import tkinter as tk
from tkinter import ttk, messagebox, font
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import random
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from transformers import pipeline, logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


logging.set_verbosity_error()


learner_data = {
    'learner_id': [1, 2, 3],
    'data_structures_score': [85, 72, 90],
    'algorithms_score': [78, 68, 65],
    'operating_systems_score': [92, 88, 80],
    'dbms_score': [75, 80, 70],
    'computer_networks_score': [82, 67, 88],
    'preferred_language': ['ta', 'hi', 'te']
}

class Theme:
    BG_COLOR = ""

    @staticmethod
    def apply_theme(root):
        style = ttk.Style()
        style.configure("Custom.TLabel", background=Theme.BG_COLOR)
        style.configure("Custom.TButton", padding=5)
        style.configure("Custom.TRadiobutton", background=Theme.BG_COLOR)

class Quiz:
    def __init__(self):
        self.questions = {
            'data_structures': {
                'easy': [
                    {
                        'question': "What is an array?",
                        'options': [
                            "A collection of elements of the same data type",
                            "A linked list of nodes",
                            "A binary tree structure",
                            "A hash table implementation"
                        ],
                        'answer': 0
                    },
                    {
                        'question': "What is a stack?",
                        'options': [
                            "FIFO data structure",
                            "LIFO data structure",
                            "Random access structure",
                            "Binary tree structure"
                        ],
                        'answer': 1
                    }
                ],
                'medium': [
                    {
                        'question': "What is the time complexity of binary search?",
                        'options': [
                            "O(n)",
                            "O(log n)",
                            "O(n^2)",
                            "O(1)"
                        ],
                        'answer': 1
                    }
                ],
                'hard': [
                    {
                        'question': "Which balancing technique is used in AVL trees?",
                        'options': [
                            "Red-Black coloring",
                            "Height balancing",
                            "Weight balancing",
                            "B-tree balancing"
                        ],
                        'answer': 1
                    }
                ]
            }
        }
        self.current_score = 0
        self.total_questions = 0

class LearningBot:
    def __init__(self, learner_id):
        self.learner_id = learner_id
        self.modules = ['data_structures', 'algorithms', 'operating_systems', 'dbms', 'computer_networks']
        self.progress = self.get_learner_progress()
        try:
            self.chatbot = pipeline("text-generation",
                                    model="gpt2",
                                    max_length=50,
                                    pad_token_id=50256)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.chatbot = None
        self.language = self.get_learner_language()
        self.session_activities = []

    def interact(self, question):
        try:
            if self.chatbot is None:
                return "Chat functionality is currently unavailable. Please try again later."


            context = "This is a computer science learning assistant. "
            prompt = context + question


            response = self.chatbot(prompt,
                                    max_length=100,
                                    num_return_sequences=1,
                                    do_sample=True,
                                    temperature=0.7,
                                    pad_token_id=50256)


            generated_text = response[0]['generated_text']

            clean_response = generated_text.replace(context, '').strip()


            if len(clean_response) < 10:
                return "I understand you're asking about " + question + ". Could you please be more specific?"

            return clean_response

        except Exception as e:
            print(f"Chat error: {e}")
            return f"I apologize, but I'm having trouble processing your question. Could you try rephrasing it?"

    def get_learner_progress(self):
        df = pd.DataFrame(learner_data)
        learner_data_row = df[df['learner_id'] == self.learner_id]
        return learner_data_row.iloc[0][
            ['data_structures_score', 'algorithms_score', 'operating_systems_score', 'dbms_score',
             'computer_networks_score']].to_dict()

    def get_learner_language(self):
        df = pd.DataFrame(learner_data)
        learner_data_row = df[df['learner_id'] == self.learner_id]
        return learner_data_row.iloc[0]['preferred_language']

    def suggest_content(self):
        weakest_module = min(self.progress, key=self.progress.get)
        suggestion = f"Suggested content for {weakest_module.replace('_', ' ').title()}: Learn basic concepts of {weakest_module.replace('_', ' ')}."
        return self.translate(suggestion)

    def adaptive_questioning(self):
        question_difficulty = "easy" if self.progress['data_structures_score'] < 70 else "medium"
        if self.progress['algorithms_score'] > 85:
            question_difficulty = "hard"
        question = f"What is the advanced concept in {random.choice(self.modules)}?"
        return f"Question difficulty: {question_difficulty}. Question: {question}"

    def personalized_feedback(self):
        feedback = []
        for module, score in self.progress.items():
            if score < 70:
                feedback.append(f"Need improvement in {module.replace('_', ' ').title()}")
            elif score < 80:
                feedback.append(f"Good progress in {module.replace('_', ' ').title()}")
            else:
                feedback.append(f"Excellent in {module.replace('_', ' ').title()}")
        return self.translate(" ".join(feedback))

    def translate(self, text):
        if self.language == 'ta':
            return f"[Tamil Translation]: {text}"
        elif self.language == 'hi':
            return f"[Hindi Translation]: {text}"
        elif self.language == 'te':
            return f"[Telugu Translation]: {text}"
        return text

class EnhancedLearningBotGUI:
    def __init__(self, root, learner_id):
        self.root = root
        self.bot = LearningBot(learner_id)
        self.create_gui()

    def create_gui(self):
        self.root.title("Enhanced Learning Bot")
        self.root.geometry("800x600")
        self.root.configure(bg=Theme.BG_COLOR)


        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)


        info_frame = ttk.LabelFrame(main_frame, text="Learner Information")
        info_frame.pack(fill="x", pady=(0, 20))

        learner_info = f"Learner ID: {self.bot.learner_id}\nPreferred Language: {self.bot.language.upper()}"
        ttk.Label(info_frame, text=learner_info).pack(padx=10, pady=10)


        progress_frame = ttk.LabelFrame(main_frame, text="Module Progress")
        progress_frame.pack(fill="x", pady=(0, 20))

        for module, score in self.bot.progress.items():
            module_name = module.replace('_', ' ').title()
            ttk.Label(progress_frame, text=f"{module_name}: {score}%").pack(padx=10, pady=5)


        actions_frame = ttk.LabelFrame(main_frame, text="Actions")
        actions_frame.pack(fill="x")


        ttk.Button(actions_frame, text="Get Content Suggestion",
                   command=self.show_suggestion).pack(fill="x", padx=10, pady=5)

        ttk.Button(actions_frame, text="Get Adaptive Question",
                   command=self.show_question).pack(fill="x", padx=10, pady=5)

        ttk.Button(actions_frame, text="Get Personalized Feedback",
                   command=self.show_feedback).pack(fill="x", padx=10, pady=5)

        ttk.Button(actions_frame, text="Start Quiz",
                   command=self.start_quiz).pack(fill="x", padx=10, pady=5)


        chat_frame = ttk.LabelFrame(main_frame, text="Chat with Bot")
        chat_frame.pack(fill="x", pady=(20, 0))
        self.chat_display = ScrolledText(chat_frame, height=10, wrap=tk.WORD)
        self.chat_display.pack(fill="x", padx=10, pady=(5, 0))

        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill="x", padx=10, pady=5)

        self.chat_input = ttk.Entry(input_frame)
        self.chat_input.pack(side=tk.LEFT, fill="x", expand=True)

        ttk.Button(input_frame, text="Send",
                   command=self.chat_with_bot).pack(side=tk.RIGHT, padx=(5, 0))


        self.chat_input.bind("<Return>", lambda e: self.chat_with_bot())

    def show_suggestion(self):
        suggestion = self.bot.suggest_content()
        messagebox.showinfo("Content Suggestion", suggestion)

    def show_question(self):
        question = self.bot.adaptive_questioning()
        messagebox.showinfo("Adaptive Question", question)

    def show_feedback(self):
        feedback = self.bot.personalized_feedback()
        messagebox.showinfo("Personalized Feedback", feedback)

    def chat_with_bot(self):
        question = self.chat_input.get().strip()
        if question:

            self.chat_display.insert(tk.END, f"You: {question}\n")


            response = self.bot.interact(question)


            self.chat_display.insert(tk.END, f"Bot: {response}\n\n")


            self.chat_input.delete(0, tk.END)


            self.chat_display.see(tk.END)

    def start_quiz(self):
        """Starts a new quiz session"""
        quiz_window = tk.Toplevel(self.root)
        quiz_window.title("Knowledge Quiz")
        quiz_window.geometry("600x400")
        quiz_window.configure(bg=Theme.BG_COLOR)

        quiz = Quiz()
        current_question = 0
        total_questions = 5
        score = 0

        def show_question():
            nonlocal current_question, score

            if current_question >= total_questions:

                for widget in quiz_window.winfo_children():
                    widget.destroy()

                result_text = f"Quiz Complete!\nYour Score: {score}/{total_questions}"
                ttk.Label(
                    quiz_window,
                    text=result_text,
                    style="Custom.TLabel",
                    font=("Helvetica", 16, "bold")
                ).pack(pady=20)

                ttk.Button(
                    quiz_window,
                    text="Close",
                    command=quiz_window.destroy,
                    style="Custom.TButton"
                ).pack(pady=10)


                self.bot.session_activities.append({
                    "type": "quiz",
                    "score": score,
                    "total": total_questions,
                    "timestamp": str(datetime.now())
                })
                return


            module = random.choice(list(quiz.questions.keys()))
            difficulty = random.choice(['easy', 'medium', 'hard'])


            question_data = random.choice(quiz.questions[module][difficulty])


            for widget in quiz_window.winfo_children():
                widget.destroy()


            ttk.Label(
                quiz_window,
                text=f"Question {current_question + 1}/{total_questions} ({difficulty.title()})",
                style="Custom.TLabel",
                font=("Helvetica", 12)
            ).pack(pady=10)


            ttk.Label(
                quiz_window,
                text=question_data['question'],
                style="Custom.TLabel",
                font=("Helvetica", 14, "bold"),
                wraplength=500
            ).pack(pady=20)


            selected_answer = tk.IntVar()


            for i, option in enumerate(question_data['options']):
                ttk.Radiobutton(
                    quiz_window,
                    text=option,
                    variable=selected_answer,
                    value=i,
                    style="Custom.TRadiobutton"
                ).pack(pady=5, padx=20, anchor='w')

            def submit_answer():
                nonlocal current_question, score
                if selected_answer.get() == question_data['answer']:
                    score += 1
                current_question += 1
                show_question()

            ttk.Button(
                quiz_window,
                text="Submit Answer",
                command=submit_answer,
                style="Custom.TButton"
            ).pack(pady=20)


        show_question()

if __name__ == "__main__":
    root = tk.Tk()
    Theme.apply_theme(root)
    app = EnhancedLearningBotGUI(root, learner_id=1)
    root.mainloop()