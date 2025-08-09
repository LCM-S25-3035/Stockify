import tkinter as tk
from tkinter import messagebox

def submit():
    try:
        # Get and validate inputs
        amount = float(amount_entry.get())
        risk = float(risk_entry.get())
        duration = int(duration_entry.get())

        if not (0 <= risk <= 1):
            raise ValueError("Risk must be between 0 and 1")

        # If all inputs are valid
        print(f"Amount: ${amount}, Risk: {risk}, Duration: {duration} years")
        messagebox.showinfo("Input Received", f"Amount: ${amount}\nRisk: {risk}\nDuration: {duration} years")
    
    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))

# Create the main window
root = tk.Tk()
root.title("Stock Investment Input")
root.geometry("320x250")

# Input fields
tk.Label(root, text="Amount ($):").pack(pady=(10, 0))
amount_entry = tk.Entry(root)
amount_entry.pack()

tk.Label(root, text="Risk (0 to 1):").pack(pady=(10, 0))
risk_entry = tk.Entry(root)
risk_entry.pack()

tk.Label(root, text="Duration (Years):").pack(pady=(10, 0))
duration_entry = tk.Entry(root)
duration_entry.pack()

# Submit button
tk.Button(root, text="Submit", command=submit).pack(pady=15)

# Run the GUI loop
root.mainloop()
