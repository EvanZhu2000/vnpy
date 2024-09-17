def send_email(user, pwd, recipient, subject, body):
    import smtplib

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except Exception as e:
        print("failed to send mail")
        print(e)

if __name__ == '__main__':
    send_email('alertevan@gmail.com','uesb oewt wdfd hcfr','evan.zhu@cashalgo.com',
               'Happy Mid-Autumn Festival',
               'A bright moon and stars twinkle and shine. Wishing you a merry Mid-Autumn Festival, bliss, and happiness.')