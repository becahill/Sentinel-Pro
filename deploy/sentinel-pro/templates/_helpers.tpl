{{- define "sentinel-pro.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "sentinel-pro.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "sentinel-pro.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "sentinel-pro.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "sentinel-pro.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sentinel-pro.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "sentinel-pro.secretName" -}}
{{- printf "%s-secrets" (include "sentinel-pro.fullname" .) -}}
{{- end -}}

{{- define "sentinel-pro.configName" -}}
{{- printf "%s-config" (include "sentinel-pro.fullname" .) -}}
{{- end -}}

{{- define "sentinel-pro.postgresHost" -}}
{{- printf "%s-postgres" (include "sentinel-pro.fullname" .) -}}
{{- end -}}

{{- define "sentinel-pro.redisHost" -}}
{{- printf "%s-redis" (include "sentinel-pro.fullname" .) -}}
{{- end -}}

{{- define "sentinel-pro.dbUrl" -}}
{{- if .Values.externalDatabaseUrl -}}
{{- .Values.externalDatabaseUrl -}}
{{- else -}}
{{- printf "postgresql+psycopg://%s:%s@%s:5432/%s" .Values.postgres.username .Values.postgres.password (include "sentinel-pro.postgresHost" .) .Values.postgres.database -}}
{{- end -}}
{{- end -}}

{{- define "sentinel-pro.redisUrl" -}}
{{- if .Values.externalRedisUrl -}}
{{- .Values.externalRedisUrl -}}
{{- else -}}
{{- printf "redis://%s:6379/0" (include "sentinel-pro.redisHost" .) -}}
{{- end -}}
{{- end -}}

{{- define "sentinel-pro.redisResultBackend" -}}
{{- if .Values.externalRedisUrl -}}
{{- regexReplaceAll "/[0-9]+$" .Values.externalRedisUrl "/1" -}}
{{- else -}}
{{- printf "redis://%s:6379/1" (include "sentinel-pro.redisHost" .) -}}
{{- end -}}
{{- end -}}
