# Generated by Django 5.1.4 on 2024-12-22 18:15

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Algorithm',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('status', models.CharField(default='ready', max_length=20)),
                ('progress', models.IntegerField(default=0)),
                ('max_iterations', models.IntegerField(default=1000)),
            ],
        ),
    ]
