import smtplib, ssl

def send_mail(lp_no):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "computersciencevoyager@gmail.com"  # Enter your address
    receiver_email = "7krishyadav@gmail.com"  # Enter receiver address
    password = "zvhqozhofqddrgzj"

    message = """\
    Subject: Challan

    Respected sir/ma'am,

    Your vehicle with the license number """ + lp_no + """has been spotted having improper pollution controls. Thus is unfit for roads 
    
    Kindly get the vehicle repaired and get a PUC certificate at earliest.

    Thank you
    AI POLLUTION INSPECTOR BOT
    """
    # This is to inform that your vehicle with license plate xxxx is not in compliance with PUC act of India, and you are thereby penalized.
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)