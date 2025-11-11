#!/bin/bash

# --- CONFIG ---
STEP_SIZE=0.1
N_STEPS_LIST=(10 25 50 100)  # Sweep over number of MCMC steps

for n_steps in "${N_STEPS_LIST[@]}"; do

  JOB_NAME="wnae-job-toy10d-mcmc-${n_steps}steps"
  OUTPUT_DIR="output/toy10d-mcmc-${n_steps}steps"

  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: axol1tl
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: tuner
          image: python:3.10-slim
          imagePullPolicy: Always
          volumeMounts:
            - name: axovol
              mountPath: /axovol
          workingDir: /axovol/wnae_10_14
          command:
            - "/bin/bash"
            - "-c"
            - |
              set -e
              echo "Installing dependencies..."
              python -m pip install --no-cache-dir -r requirements.txt
              python -m pip install numpy==1.23.5 scipy==1.13.1

              echo "Running training for D=10, n_steps=${n_steps}..."
              python -u train_toy.py --override \
                data.N=3 \
                data.D=10 \
                training.n_epochs=500 \
                wnae.x_step_size=${STEP_SIZE} \
                wnae.x_step=${n_steps} \
                data.min_max=true \
                data.output="${OUTPUT_DIR}" \
                | tee train_${JOB_NAME}.log
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: 1
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: 1
      volumes:
        - name: axovol
          persistentVolumeClaim:
            claimName: axovol
EOF

done
