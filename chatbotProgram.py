import customtkinter as ctk
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai

# Configure API and model
genai.configure(api_key='SECRET_KEY')
model = 'models/embedding-001'

# Predefined responses for the first bot
responses = [
    "You can buy tickets on our website or at the station.",
    "The bus schedule is from 6 a.m. to 10 p.m.",
    "The bus fare depends on your destination.",
    "Our buses depart every 30 minutes.",
    "You can cancel your ticket 24 hours before departure.",
    "Please provide an ID document when purchasing your ticket.",
    "Children under 5 years old travel for free.",
    "You can bring up to two bags in the luggage compartment at no extra cost.",
    "The buses are equipped with free Wi-Fi.",
    "We have special services for people with disabilities.",
    "No, no vuelva con su ex, es mejor seguir adelante.",
    
]

# Convert responses to embeddings using the Gemini API
response_embeddings = genai.embed_content(model=model,
                                          content=responses,
                                          task_type="retrieval_document")

def find_similar_response(question, threshold=0.8):
    global response_embeddings
    question_embedding = genai.embed_content(model=model,
                                             content=question,
                                             task_type="retrieval_document")
    question_embeddingArray = np.array(question_embedding["embedding"]).reshape(1, -1)
    response_embeddingsArray = np.array(response_embeddings["embedding"])
    # Calculate cosine similarity between the question and each response in dictionary
    similarities = cosine_similarity(question_embeddingArray, response_embeddingsArray)
    similarities = similarities.flatten()  # Flatten the similarities array
    # Get the index of the most similar response
    index = np.argmax(similarities)
    max_similarity = similarities[index]
    # Check if the highest similarity score is below the threshold
    if max_similarity < threshold:
        return "I'm sorry, I don't understand your question. Can you rephrase it?"
    return responses[index]

def display_message(text, is_user):
    message_frame = ctk.CTkFrame(chat_history, corner_radius=10, fg_color="#1F538D" if is_user else "#343638")
    message_label = ctk.CTkLabel(message_frame, text=text, wraplength=400, font=("Arial", 12))
    message_label.pack(padx=10, pady=10)
    message_frame.pack(anchor="e" if is_user else "w", padx=10, pady=5)

def send():
    question = user_input.get()
    if question.strip() != "":
        display_message(question, True)
        response = find_similar_response(question)
        display_message(response, False)
        user_input.delete(0, ctk.END)

# Create the main window
root = ctk.CTk()
root.title("Chatbot")

# Set dark mode (you can change to 'light' or 'system' based on preference)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Create the header panel
header_frame = ctk.CTkFrame(root, width=600, height=80, fg_color="#1F538D")
header_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=10)

# Add the bot's profile picture
profile_image = Image.open("profile.jpg")
profile_image = profile_image.resize((50, 50), resample=Image.LANCZOS)
profile_photo = ImageTk.PhotoImage(profile_image)
profile_label = ctk.CTkLabel(header_frame, image=profile_photo, text="", anchor="w")
profile_label.place(x=10, y=15)

# Add the bot's name
bot_name_label = ctk.CTkLabel(header_frame, text="Buserks Chatbot", font=("Arial", 20, "bold"))
bot_name_label.place(x=70, y=20)

# Create a scrolled text widget for chat history
chat_history = ctk.CTkScrollableFrame(root, width=600, height=400)
chat_history.grid(row=1, column=0, columnspan=2, padx=20, pady=20)

# Create an entry widget for the user to type their question
user_input = ctk.CTkEntry(root, width=400, placeholder_text="Type your question here...")
user_input.grid(row=2, column=0, pady=10, padx=10)

# Create a button to send the question
send_button = ctk.CTkButton(root, text="Send", command=send)
send_button.grid(row=2, column=1, pady=10, padx=10)

root.mainloop()