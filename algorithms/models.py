from django.db import models


class Algorithm(models.Model):
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=20,
                              default="ready")  # Możliwe wartości: ready, running, paused, stopped, completed
    progress = models.IntegerField(default=0)  # Procent ukończenia
    max_iterations = models.IntegerField(default=1000)  # Maksymalna liczba iteracji

    def __str__(self):
        return self.name
