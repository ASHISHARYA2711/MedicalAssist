# Generated by Django 4.2.1 on 2023-06-02 01:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('appointment', '0002_appointment_patientdischargedetails_patient'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='patientdischargedetails',
            name='roomCharge',
        ),
    ]
