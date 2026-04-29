create extension if not exists pgcrypto;

create table if not exists public.app_users (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  client_id text unique,
  email text,
  display_name text
);

create table if not exists public.analysis_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  client_id text,
  user_id uuid references public.app_users(id) on delete set null,
  acne_severity text not null,
  acne_score double precision not null,
  lesion_count integer not null,
  zone_counts jsonb not null,
  hyperpigmentation jsonb not null,
  summary text not null,
  annotated_image_base64 text not null,
  heatmap_image_base64 text not null
);

create table if not exists public.detailed_reports (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  client_id text,
  user_id uuid references public.app_users(id) on delete set null,
  analysis_summary text not null,
  acne_severity text not null,
  model text not null,
  generated_by text not null,
  report_text text not null,
  disclaimer text not null
);

create table if not exists public.progress_runs (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  client_id text,
  user_id uuid references public.app_users(id) on delete set null,
  similarity double precision not null,
  baseline_lesions integer not null,
  followup_lesions integer not null,
  improvement_percent double precision not null,
  timeline text not null,
  summary text not null,
  stages jsonb not null
);

create index if not exists idx_analysis_runs_client_id_created_at
  on public.analysis_runs(client_id, created_at desc);
create index if not exists idx_detailed_reports_client_id_created_at
  on public.detailed_reports(client_id, created_at desc);
create index if not exists idx_progress_runs_client_id_created_at
  on public.progress_runs(client_id, created_at desc);

create or replace view public.v_client_latest_summary as
select
  ar.client_id,
  ar.created_at,
  ar.acne_severity,
  ar.acne_score,
  ar.lesion_count,
  ar.summary
from public.analysis_runs ar
where ar.client_id is not null;

alter table public.app_users enable row level security;
alter table public.analysis_runs enable row level security;
alter table public.detailed_reports enable row level security;
alter table public.progress_runs enable row level security;

drop policy if exists "service_role_insert_users" on public.app_users;
create policy "service_role_insert_users"
on public.app_users
for insert
to service_role
with check (true);

drop policy if exists "service_role_select_users" on public.app_users;
create policy "service_role_select_users"
on public.app_users
for select
to service_role
using (true);

drop policy if exists "service_role_insert_analysis" on public.analysis_runs;
create policy "service_role_insert_analysis"
on public.analysis_runs
for insert
to service_role
with check (true);

drop policy if exists "service_role_select_analysis" on public.analysis_runs;
create policy "service_role_select_analysis"
on public.analysis_runs
for select
to service_role
using (true);

drop policy if exists "service_role_insert_reports" on public.detailed_reports;
create policy "service_role_insert_reports"
on public.detailed_reports
for insert
to service_role
with check (true);

drop policy if exists "service_role_select_reports" on public.detailed_reports;
create policy "service_role_select_reports"
on public.detailed_reports
for select
to service_role
using (true);

drop policy if exists "service_role_insert_progress" on public.progress_runs;
create policy "service_role_insert_progress"
on public.progress_runs
for insert
to service_role
with check (true);

drop policy if exists "service_role_select_progress" on public.progress_runs;
create policy "service_role_select_progress"
on public.progress_runs
for select
to service_role
using (true);
