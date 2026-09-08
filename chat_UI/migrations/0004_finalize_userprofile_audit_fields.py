from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("chat_UI", "0003_userprofile_audit_fields")]

    operations = [
        migrations.AlterField(
            model_name="userprofile",
            name="created_at",
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name="userprofile",
            name="updated_at",
            field=models.DateTimeField(auto_now=True),
        ),
    ]
