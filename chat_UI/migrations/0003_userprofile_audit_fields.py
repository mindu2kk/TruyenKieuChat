# Generated manually to preserve existing accounts when moving to PostgreSQL.

from django.conf import settings
from django.db import migrations, models
from django.utils import timezone


def create_missing_profiles(apps, schema_editor):
    User = apps.get_model("auth", "User")
    UserProfile = apps.get_model("chat_UI", "UserProfile")
    existing_user_ids = set(UserProfile.objects.values_list("user_id", flat=True))
    UserProfile.objects.bulk_create(
        [UserProfile(user_id=user_id) for user_id in User.objects.exclude(pk__in=existing_user_ids).values_list("pk", flat=True)]
    )


class Migration(migrations.Migration):
    dependencies = [
        ("chat_UI", "0002_chatmessage_meta"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AddField(
            model_name="userprofile",
            name="created_at",
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name="userprofile",
            name="updated_at",
            field=models.DateTimeField(null=True),
        ),
        migrations.RunPython(create_missing_profiles, migrations.RunPython.noop),
        migrations.RunSQL(
            'UPDATE "chat_UI_userprofile" SET created_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP '
            "WHERE created_at IS NULL OR updated_at IS NULL",
            migrations.RunSQL.noop,
        ),
    ]
