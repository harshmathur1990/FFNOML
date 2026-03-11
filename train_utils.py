import torch
from tqdm import tqdm


def extract_temperature(X):
    """
    X: (B, C, Nz, H, W)
       channel 0 is log10(T)

    returns:
       T: (B, Nz, H, W)
    """
    logT = X[:, 0]
    T = 10.0 ** logT
    return T


def compute_loss(pred, target, weight, loss_fn, T):

    loss, components = loss_fn(T, pred, target)

    if weight is not None:
        loss = loss * weight

    loss = loss.mean()

    return loss, components


def flatten_columns_logb(logb):
    """
    logb: (B, L, Nz, Ny, Nx)
    -> (B*Ny*Nx, L, Nz)
    """
    B, L, Nz, Ny, Nx = logb.shape
    return logb.permute(0,3,4,1,2).reshape(B*Ny*Nx, L, Nz)


def flatten_columns_T(T):
    """
    T: (B, Nz, Ny, Nx)
    -> (B*Ny*Nx, Nz)
    """
    B, Nz, Ny, Nx = T.shape
    return T.permute(0,2,3,1).reshape(B*Ny*Nx, Nz)


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device,
    epoch,
    num_epochs,
    amp=True,
    grad_clip=1.0,
):

    model.train()
    running = 0.0

    lr = optimizer.param_groups[0]["lr"]

    pbar = tqdm(loader, desc=f"epoch {epoch}/{num_epochs}", leave=False)

    for it, (x, y, dx, dy, scale, weight) in enumerate(pbar, start=1):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        dx = dx.to(device, non_blocking=True)
        dy = dy.to(device, non_blocking=True)

        if weight is not None:
            weight = weight.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.startswith("cuda"))):

            pred = model(x, dx, dy)

            T = extract_temperature(x)

            pred = flatten_columns_logb(pred)
            y = flatten_columns_logb(y)
            T = flatten_columns_T(T)

            loss, components = compute_loss(
                pred=pred,
                target=y,
                weight=weight,
                loss_fn=loss_fn,
                T=T,
            )

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        running += loss.item()

        # ---------------------------
        # build tqdm metrics
        # ---------------------------

        postfix = {
            "loss": f"{loss.item():.2e}",
            "lr": f"{lr:.1e}",
        }

        if device.startswith("cuda"):
            mem = torch.cuda.memory_allocated() / 1024**3
            postfix["gpu_mem"] = f"{mem:.2f}G"

        if components is not None and isinstance(components, dict):

            if "data" in components:
                postfix["L1"] = f"{components['data'].item():.2e}"

            if "source" in components:
                postfix["L2"] = f"{components['source'].item():.2e}"

            if "source_per_atom" in components and "atom_names" in components:

                spa = components["source_per_atom"]
                names = components["atom_names"]

                for j, n in enumerate(names):
                    postfix[n] = f"{spa[j].item():.2e}"

        pbar.set_postfix(postfix)

    return running / max(1, len(loader))


def validate(
    model,
    loader,
    loss_fn,
    device,
    amp=True,
):

    model.eval()

    tot = 0.0
    n = 0

    pbar = tqdm(loader, desc="val", leave=False)

    with torch.no_grad():

        for x, y, dx, dy, scale, weight in pbar:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            dx = dx.to(device, non_blocking=True)
            dy = dy.to(device, non_blocking=True)

            if weight is not None:
                weight = weight.to(device, non_blocking=True)

            T = extract_temperature(x)

            with torch.amp.autocast("cuda", enabled=(amp and device.startswith("cuda"))):

                pred = model(x, dx, dy)

                pred = flatten_columns_logb(pred)
                y = flatten_columns_logb(y)
                T = flatten_columns_T(T)

                loss, _ = compute_loss(
                    pred=pred,
                    target=y,
                    weight=weight,
                    loss_fn=loss_fn,
                    T=T,
                )

            tot += loss.item()
            n += 1

            pbar.set_postfix(loss=f"{loss.item():.2e}")

    return tot / max(1, n)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    scaler,
    save_path,
    *,
    num_epochs=50,
    device="cuda",
    amp=True,
    grad_clip=1.0,
    early_stopping=None,
):

    best_val = float("inf")

    if early_stopping is not None:
        es_enabled = early_stopping.get("enabled", False)
        patience = early_stopping.get("patience", 10)
        min_delta = early_stopping.get("min_delta", 0.0)
    else:
        es_enabled = False
        patience = 0
        min_delta = 0.0

    epochs_no_improve = 0

    torch.cuda.empty_cache()

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            amp=amp,
            grad_clip=grad_clip,
        )

        if val_loader is not None:

            val_loss = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                amp=amp,
            )

            improved = (best_val - val_loss) > min_delta

            if improved:

                best_val = val_loss
                epochs_no_improve = 0

                torch.save(
                    dict(
                        epoch=epoch,
                        model_state=model.state_dict(),
                        opt_state=optimizer.state_dict(),
                        train_loss=train_loss,
                        val_loss=val_loss,
                    ),
                    save_path,
                )

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train={train_loss:.6e} "
                    f"val={val_loss:.6e} (saved best)"
                )

            else:

                epochs_no_improve += 1

                print(
                    f"[Epoch {epoch:03d}] "
                    f"train={train_loss:.6e} "
                    f"val={val_loss:.6e}"
                )

                if es_enabled and epochs_no_improve > patience:
                    print(f"Early stopping triggered (patience={patience})")
                    break

        else:

            torch.save(
                dict(
                    epoch=epoch,
                    model_state=model.state_dict(),
                    opt_state=optimizer.state_dict(),
                    train_loss=train_loss,
                ),
                save_path,
            )

            print(f"[Epoch {epoch:03d}] train={train_loss:.6e} (saved)")
